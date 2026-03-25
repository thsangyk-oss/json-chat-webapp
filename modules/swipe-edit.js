'use strict';

/**
 * swipe-edit.js — Swipe/Regenerate + Edit Message module
 *
 * Exports: register(ctx) → { 'POST /api/swipe': handler, 'POST /api/edit-message': handler }
 *
 * ctx: { readState, writeState, readBrain, chatCompletion, checkAuth,
 *         json, readBody, buildPromptMessages, getContextAiSettings }
 */

module.exports = function register(ctx) {
  const {
    readState,
    writeState,
    checkAuth,
    json,
    readBody,
    chatCompletion,
    buildPromptMessages,
    getContextAiSettings,
  } = ctx;

  // ── helpers ────────────────────────────────────────────────────────────────

  /** Ensure activeSessionId is valid and return the session object */
  function getActiveSession(state) {
    const s = state.sessions && state.sessions[state.activeSessionId];
    if (!s) throw new Error('No active session');
    return s;
  }

  /** Wrap index within [0, len) */
  function wrap(idx, len) {
    if (len === 0) return 0;
    return ((idx % len) + len) % len;
  }

  /**
   * Build a minimal prompt for regeneration given the full session and the
   * user message that should be the last human turn.
   *
   * We replicate only what /api/chat does (blocks + history + userMsg) but
   * without side-effects (no memory extraction, no summarise trigger).
   */
  function buildRegenPrompt(state, session, userMsg, ai) {
    // Build a stripped-down blocks object — we don't have access to all the
    // helpers here, so we delegate to buildPromptMessages with minimal system.
    // If the caller injected buildPromptBlocks we could use it; otherwise we
    // construct a shallow prompt that still works.

    // History up to (but not including) the user message we are regenerating
    // from — caller already trimmed session.messages before calling this.
    const maxHistory = ai.contextWindowMode === '500_messages' ? 500 : 140;
    const history = (session.messages || []).slice(-maxHistory).map(m => ({
      role: m.role,
      content: m.alternatives
        ? (m.alternatives[m.activeIndex != null ? m.activeIndex : 0] || m.content || '')
        : (m.content || ''),
    }));

    const userMsgObj = { role: 'user', content: userMsg };

    // Use buildPromptMessages if we have blocks; otherwise fall back to a
    // plain array so chatCompletion always gets a valid messages array.
    let msgs;
    if (typeof ctx.buildPromptBlocks === 'function') {
      // Bonus: if the main server exposed buildPromptBlocks we can use it
      try {
        const loadContext = ctx.loadContext;
        const ctxCard = loadContext ? loadContext(state.activeContext) : null;
        const blocks = ctxCard
          ? ctx.buildPromptBlocks(ctxCard, ai, session, '', '', '')
          : { system: '' };
        msgs = buildPromptMessages(blocks, ai, history, userMsgObj);
      } catch {
        msgs = [...history, userMsgObj];
      }
    } else {
      msgs = [...history, userMsgObj];
    }

    return msgs;
  }

  // ── POST /api/swipe ────────────────────────────────────────────────────────

  async function handleSwipe(req, res, parsed) {
    if (!checkAuth(req, parsed)) {
      return json(res, 401, { error: 'Unauthorized' });
    }

    let body;
    try {
      body = await readBody(req);
    } catch (e) {
      return json(res, 400, { error: 'Invalid JSON body' });
    }

    const direction = body.direction; // "next" | "prev" | "new"
    if (!['next', 'prev', 'new'].includes(direction)) {
      return json(res, 400, { error: 'direction must be "next", "prev", or "new"' });
    }

    const state = readState();
    let session;
    try {
      session = getActiveSession(state);
    } catch (e) {
      return json(res, 404, { error: e.message });
    }

    const msgs = session.messages || [];

    // Find the last assistant message
    let lastAssistantIdx = -1;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === 'assistant') { lastAssistantIdx = i; break; }
    }

    if (lastAssistantIdx === -1) {
      return json(res, 400, { error: 'No assistant message to swipe' });
    }

    const assistantMsg = msgs[lastAssistantIdx];

    // Ensure alternatives array exists (migrate legacy messages on the fly)
    if (!Array.isArray(assistantMsg.alternatives) || assistantMsg.alternatives.length === 0) {
      assistantMsg.alternatives = [assistantMsg.content || ''];
      assistantMsg.activeIndex = 0;
    }
    if (assistantMsg.activeIndex == null) assistantMsg.activeIndex = 0;

    if (direction === 'next') {
      assistantMsg.activeIndex = wrap(assistantMsg.activeIndex + 1, assistantMsg.alternatives.length);
      assistantMsg.content = assistantMsg.alternatives[assistantMsg.activeIndex];

    } else if (direction === 'prev') {
      assistantMsg.activeIndex = wrap(assistantMsg.activeIndex - 1, assistantMsg.alternatives.length);
      assistantMsg.content = assistantMsg.alternatives[assistantMsg.activeIndex];

    } else {
      // "new" — regenerate: find user message that precedes the last assistant msg
      let userMsgText = '';
      for (let i = lastAssistantIdx - 1; i >= 0; i--) {
        if (msgs[i].role === 'user') { userMsgText = msgs[i].content || ''; break; }
      }

      if (!userMsgText) {
        return json(res, 400, { error: 'No preceding user message found' });
      }

      // Build context for LLM — use messages up to (not including) last assistant
      const ai = getContextAiSettings(state, state.activeContext);
      const sessionForPrompt = {
        ...session,
        messages: msgs.slice(0, lastAssistantIdx),
      };

      let promptMsgs;
      try {
        promptMsgs = buildRegenPrompt(state, sessionForPrompt, userMsgText, ai);
      } catch (e) {
        return json(res, 500, { error: `Prompt build failed: ${e.message}` });
      }

      let assistant;
      try {
        assistant = await chatCompletion(promptMsgs, ai);
      } catch (e) {
        return json(res, 502, { error: `LLM error: ${e.message}` });
      }

      const newReply = (assistant && assistant.content) || '';
      assistantMsg.alternatives.push(newReply);
      assistantMsg.activeIndex = assistantMsg.alternatives.length - 1;
      assistantMsg.content = newReply;
    }

    // Persist
    session.messages = msgs;
    writeState(state);

    return json(res, 200, {
      reply: assistantMsg.content,
      index: assistantMsg.activeIndex,
      total: assistantMsg.alternatives.length,
    });
  }

  // ── POST /api/edit-message ─────────────────────────────────────────────────

  async function handleEditMessage(req, res, parsed) {
    if (!checkAuth(req, parsed)) {
      return json(res, 401, { error: 'Unauthorized' });
    }

    let body;
    try {
      body = await readBody(req);
    } catch (e) {
      return json(res, 400, { error: 'Invalid JSON body' });
    }

    const { index, content, regenerate } = body;

    if (typeof index !== 'number' || !Number.isInteger(index) || index < 0) {
      return json(res, 400, { error: 'index must be a non-negative integer' });
    }
    if (typeof content !== 'string') {
      return json(res, 400, { error: 'content must be a string' });
    }

    const state = readState();
    let session;
    try {
      session = getActiveSession(state);
    } catch (e) {
      return json(res, 404, { error: e.message });
    }

    const msgs = session.messages || [];

    if (index >= msgs.length) {
      return json(res, 400, { error: `index ${index} out of range (${msgs.length} messages)` });
    }

    const targetMsg = msgs[index];

    // Update the message content
    targetMsg.content = content;

    if (targetMsg.role === 'assistant') {
      // Keep alternatives in sync with the active slot
      if (!Array.isArray(targetMsg.alternatives) || targetMsg.alternatives.length === 0) {
        targetMsg.alternatives = [content];
        targetMsg.activeIndex = 0;
      } else {
        const ai = targetMsg.activeIndex != null ? targetMsg.activeIndex : 0;
        targetMsg.activeIndex = ai;
        targetMsg.alternatives[ai] = content;
      }
    }

    // Regenerate: only for user messages when regenerate=true
    if (regenerate && targetMsg.role === 'user') {
      // Truncate everything after this user message
      session.messages = msgs.slice(0, index + 1);

      const ai = getContextAiSettings(state, state.activeContext);
      const sessionForPrompt = {
        ...session,
        messages: session.messages.slice(0, index), // history before the user msg
      };

      let promptMsgs;
      try {
        promptMsgs = buildRegenPrompt(state, sessionForPrompt, content, ai);
      } catch (e) {
        return json(res, 500, { error: `Prompt build failed: ${e.message}` });
      }

      let assistant;
      try {
        assistant = await chatCompletion(promptMsgs, ai);
      } catch (e) {
        return json(res, 502, { error: `LLM error: ${e.message}` });
      }

      const newReply = (assistant && assistant.content) || '';
      const assistantMsgStored = {
        role: 'assistant',
        content: newReply,
        alternatives: [newReply],
        activeIndex: 0,
      };

      session.messages.push(assistantMsgStored);
      writeState(state);

      return json(res, 200, {
        reply: newReply,
        messages: session.messages,
      });
    }

    // No regeneration — just save and return updated messages
    writeState(state);

    return json(res, 200, {
      messages: session.messages,
    });
  }

  // ── Route table ───────────────────────────────────────────────────────────

  return {
    'POST /api/swipe': handleSwipe,
    'POST /api/edit-message': handleEditMessage,
  };
};
