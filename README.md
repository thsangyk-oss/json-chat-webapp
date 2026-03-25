# JSON Context Chat WebApp (mobile-first)

Runs on port **8877**.

## Features
- Chat with bot using active JSON context
- Switch context JSON anytime (`/contexts/*.json`)
- Persistent conversation state in `data/state.json`
- Mobile-friendly UI
- Per-context long-term memory (Memory V2):
  - `data/memory/<context>/profile.json`
  - `data/memory/<context>/state.json`
  - `data/memory/<context>/episodes.ndjson`
  - `data/memory/<context>/summary.md`
  - plus editable `data/memory/<context>.memory.md`

## Run

```bash
cd /root/.openclaw/workspace/json-chat-webapp
node server.js
```

Open: `http://<host-ip>:8877`

## Notes
- Default model: `CLIProxyAPI/gpt-5.4-mini`
- Set model env if needed:

```bash
JSON_CHAT_MODEL="CLIProxyAPI/gpt-5.4" node server.js
```

- Add new contexts by placing JSON files into `contexts/`.
- Supports `chara_card_v2` format and simple custom JSON.
