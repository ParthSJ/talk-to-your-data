# Person A setup — Talk to Your Data

This is the real [AssemblyAI voice-agent-starter-python](https://github.com/AssemblyAI/voice-agent-starter-python) repo, with two files added: `agents/talk-to-your-data.jsonc` (validated against the repo's own JSONC parser) and `mock_server.py` (a stand-in for Person B's real service, tested below).

## What's in it

- **Persona + dialogue**: the system prompt encodes the demo script directly — mandatory `get_dataset_context` first, `age`/`bmi`/`smoker` required before calling `predict`, the exact reliability wording spoken as given (never invented), and "state the assumption" for any impact claim beyond a single prediction.
- **Voice**: `anna` (a real ID from AssemblyAI's catalog — swap it for another from [the voice list](https://www.assemblyai.com/docs/voice-agents/voice-agent-api/voices) if you want a different feel, just don't invent an ID that isn't in that catalog).
- **LLM**: Claude through AssemblyAI's own gateway, so your `ASSEMBLYAI_API_KEY` is the only credential needed — no separate Anthropic key.
- **Keyterms**: biased toward BMI, smoker/non-smoker, the four US regions, etc. — so the transcriber doesn't mishear domain words.
- **Turn detection**: slightly more patient (`min_silence: 900`) than default, since callers pause mid-sentence working out a hypothetical case's numbers.
- **Three tools**: `get_dataset_context`, `run_analysis`, `predict` — matching the team's tool contract exactly (see the project doc, "The tool contract — build against this, separately"). Each points at `${BASE_URL}/tools/...`.

## Testing before Person B's real service exists

`mock_server.py` (repo root) answers all three tools to the exact contract shape — standard library only, nothing to install. It's a testing aid, not Person B's real work: `predict` uses a made-up formula, not a trained model, clearly labeled as such in the file.

```sh
python3 mock_server.py        # serves on :8000
```

Then tunnel it so AssemblyAI's servers can reach it (they call your tools over the public internet, not localhost):

```sh
ngrok http 8000
# or: cloudflared tunnel --url http://localhost:8000
```

Verified against the full contract, including error paths:

```sh
curl -s -X POST http://localhost:8000/tools/get_dataset_context
curl -s -X POST http://localhost:8000/tools/run_analysis
curl -s -X POST http://localhost:8000/tools/predict -d '{"age":35,"bmi":28,"smoker":"yes"}'
curl -s -X POST http://localhost:8000/tools/predict -d '{"age":35,"bmi":28}'   # missing smoker -> ok:false
```

Swap `BASE_URL` to Person B's real service the moment it exists — nothing on the voice side changes, because the JSON shape is identical.

## What you need to do

1. `cp .env.example .env`, then add:
   ```
   ASSEMBLYAI_API_KEY=your_key_here      # https://www.assemblyai.com/dashboard/api-keys
   BASE_URL=https://your-tunnel-url.app  # mock_server.py's tunnel URL for now, Person B's later — no trailing slash
   ```

2. Publish:
   ```
   AGENT=talk-to-your-data python publish.py
   ```

3. Talk to it:
   ```
   python deployment/browser/server.py
   ```
   Open `http://localhost:3000`.

4. Whenever `BASE_URL` changes (tunnel restarts — free tiers rotate the URL every time), update `.env` and re-run `AGENT=talk-to-your-data python publish.py`. Publishing again updates this same agent rather than creating a new one.

## Testing against the real dialogue script

Once `BASE_URL` points at Person B's real endpoints:

1. "What features are associated with higher insurance charges?"
2. Describe a hypothetical person, including age, BMI, and smoker status — if you leave one out on purpose, confirm the agent asks for it instead of guessing.
3. "How reliable is that?" — confirm it repeats the tool's reliability sentence rather than inventing one.
4. "Explain that like I'm a business manager."

If any of these feels off, the fix is almost always the `system_prompt` in `agents/talk-to-your-data.jsonc`, not the tool code — that's the boundary the team agreed on.
