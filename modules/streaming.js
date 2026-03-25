const fs = require('fs');
const path = require('path');

module.exports = function register(ctx) {
  return {
    'POST /api/chat-stream': async (req, res, parsed) => {
      if (!ctx.checkAuth(req, parsed)) {
        res.writeHead(401, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ error: 'Unauthorized' }));
      }

      let body;
      try { body = await ctx.readBody(req); } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ error: e.message }));
      }

      const userText = (body.message || '').trim();
      if (!userText) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ error: 'message required' }));
      }

      // SSE headers
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
      });

      try {
        const state = ctx.readState();
        const ctxName = state.activeContext;
        if (!ctxName) {
          res.write(`data: ${JSON.stringify({ error: 'No active context' })}\n\n`);
          return res.end();
        }

        const ai = ctx.getContextAiSettings(state, ctxName);
        const session = state.sessions?.[ctxName] || { messages: [], summaries: [] };
        if (!state.sessions) state.sessions = {};
        state.sessions[ctxName] = session;

        // Add user message
        const userMsg = { role: 'user', content: userText };
        session.messages.push(userMsg);

        // Build prompt (lorebook, recall, memory, NSFW, author's note, etc)
        const msgs = ctx.buildPromptMessages(ctxName, session, ai, userText);

        // Read brain config
        const brain = ctx.readBrain();

        // Build request body
        const reqBody = {
          model: brain.activeModel || brain.model || 'gpt-5.4',
          messages: msgs,
          temperature: ctx.getTemperature ? ctx.getTemperature(ai) : 0.8,
          stream: true,
        };

        if (ai.reasoning === 'on') {
          reqBody.reasoning = { effort: 'high' };
        }

        // Determine max tokens
        const targetWords = ai.targetWords || 180;
        const maxTokens = Math.max(400, Math.round(targetWords * 2.5));
        if (ai.reasoning === 'on') {
          reqBody.max_completion_tokens = maxTokens + 8000;
        } else {
          reqBody.max_completion_tokens = maxTokens;
        }

        // Get active provider URL
        let baseUrl = brain.baseUrl;
        let apiKey = brain.apiKey;
        if (brain.providers && brain.providers.length > 0) {
          const prov = brain.providers[brain.activeProvider || 0];
          if (prov) {
            baseUrl = prov.baseUrl;
            apiKey = prov.apiKey;
          }
        }

        const response = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`,
          },
          body: JSON.stringify(reqBody),
        });

        if (!response.ok) {
          const err = await response.text();
          res.write(`data: ${JSON.stringify({ error: `LLM error: ${response.status}`, details: err.slice(0, 200) })}\n\n`);
          return res.end();
        }

        // Parse SSE stream from OpenAI
        let fullReply = '';
        let fullReasoning = '';
        const reader = response.body;

        // Node.js readable stream
        let buffer = '';
        for await (const chunk of reader) {
          buffer += (typeof chunk === 'string' ? chunk : new TextDecoder().decode(chunk));

          // Process complete lines
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // keep incomplete line

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data: ')) continue;
            const payload = trimmed.slice(6);
            if (payload === '[DONE]') continue;

            try {
              const parsed = JSON.parse(payload);
              const delta = parsed.choices?.[0]?.delta;
              if (!delta) continue;

              if (delta.content) {
                fullReply += delta.content;
                res.write(`data: ${JSON.stringify({ token: delta.content, done: false })}\n\n`);
              }
              if (delta.reasoning_content) {
                fullReasoning += delta.reasoning_content;
                res.write(`data: ${JSON.stringify({ reasoning: delta.reasoning_content, done: false })}\n\n`);
              }
            } catch (e) { /* skip bad JSON */ }
          }
        }

        // Process remaining buffer
        if (buffer.trim()) {
          const trimmed = buffer.trim();
          if (trimmed.startsWith('data: ') && trimmed.slice(6) !== '[DONE]') {
            try {
              const parsed = JSON.parse(trimmed.slice(6));
              const delta = parsed.choices?.[0]?.delta;
              if (delta?.content) {
                fullReply += delta.content;
                res.write(`data: ${JSON.stringify({ token: delta.content, done: false })}\n\n`);
              }
            } catch (e) {}
          }
        }

        // Final event
        res.write(`data: ${JSON.stringify({
          token: '',
          done: true,
          fullReply,
          reasoning: fullReasoning || undefined,
        })}\n\n`);
        res.end();

        // Save to session (async, non-blocking)
        const assistantMsg = {
          role: 'assistant',
          content: fullReply,
          alternatives: [fullReply],
          activeIndex: 0,
        };
        if (fullReasoning) assistantMsg.reasoning = fullReasoning;
        session.messages.push(assistantMsg);
        session.updatedAt = new Date().toISOString();
        ctx.writeState(state);

        // Memory extraction + summarize (fire-and-forget)
        if (ctx.llmExtractMemory) {
          ctx.llmExtractMemory(userText, fullReply, ctxName, ai).then(extracted => {
            if (extracted && ctx.applyLlmMemory) {
              ctx.applyLlmMemory(ctxName, userText, fullReply, extracted);
            }
          }).catch(() => {});
        }
        if (ctx.summarizeOldMessages) {
          ctx.summarizeOldMessages(session, ai).then(() => {
            ctx.writeState(ctx.readState()); // re-read to avoid overwrite
          }).catch(() => {});
        }

      } catch (err) {
        try {
          res.write(`data: ${JSON.stringify({ error: err.message, done: true })}\n\n`);
          res.end();
        } catch (e) {}
      }
    },
  };
};
