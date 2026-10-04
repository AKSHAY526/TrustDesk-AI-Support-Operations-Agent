# TrustDesk: AI Support Operations Agent

A support agent built for the [Airtribe TrustDesk capstone](https://github.com/airtribe-projects/trustdesk-capstone).

It triages incoming tickets, finds the relevant policy, and drafts a reply that cites the policy it used. It can also suggest an action, such as a replacement order or a refund review, but anything involving money or account access waits for a manager to approve it. Every AI run is logged, so a reply can be traced back to what the model saw.

The model only proposes. Whether an action is allowed, whether it needs approval, and whether it has already run are decided in code and enforced by the database, so a model that has been talked into something still can't act on it.

## Features

- Hybrid policy search (SQLite FTS5 plus sentence embeddings) over policy sections
- Triage into category, priority, sentiment and escalation
- Draft replies with citations, checked against what search actually returned
- All six tools from `tool_actions.json`, with the catalogue's rules enforced: allowed categories, amount caps and approval
- Agent, manager and admin roles. Each role sees only the console panels it can use, and the API refuses the rest. Only a manager or admin can approve an action, and never one they requested
- Idempotent actions, so retries and double clicks don't create duplicates
- Prompt injection screening, including instructions hidden in policy documents
- A trace for every AI run, plus a report of escalation reasons
- An eval runner for the provided cases
- A simple web console

## Run with Docker

The image installs everything, downloads the embedding model and seeds the database, so the only thing you need is [Docker Desktop](https://www.docker.com/products/docker-desktop/), or Docker Engine with Compose 2.24 or later.

```
git clone https://github.com/AKSHAY526/TrustDesk-AI-Support-Operations-Agent.git
cd TrustDesk-AI-Support-Operations-Agent
docker compose up --build
```

The first build takes several minutes, mostly for CPU-only PyTorch and the embedding model. When the log shows `Application startup complete`, open http://localhost:8000/ for the console or http://localhost:8000/docs for the API reference.

Without a `.env` file the container uses the offline stand-in model. To use a real model, create `.env` from `.env.example` before starting (see [Choosing a model](#choosing-a-model)). For Ollama running on your own machine, use:

```ini
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2
LLM_TIMEOUT_SECONDS=180
```

Inside the container `localhost` is the container itself, and `host.docker.internal` is how it reaches your machine. On Linux, also start Ollama with `OLLAMA_HOST=0.0.0.0` so it accepts connections from the container. `.env` is passed in when the container starts and is never copied into the image.

Everyday commands, from a second terminal in the project folder:

| Task | Command |
|---|---|
| Run the tests | `docker compose run --rm trustdesk pytest` |
| Run the evals | `docker compose exec trustdesk python -m scripts.run_evals` |
| Reset the demo data | `docker compose exec trustdesk python -m scripts.seed --reset` |
| Apply a changed `.env` | `docker compose up --force-recreate` |
| Stop | Ctrl+C, or `docker compose down` |

The database lives inside the container. `docker compose stop` keeps it for the next start, while `docker compose down` and `--force-recreate` start again from fresh seed data.

| Problem | Fix |
|---|---|
| `port is already allocated` | Something else is using port 8000, perhaps the server from the steps below. Stop it, or change the ports line in `docker-compose.yml` to `"8001:8000"` and open port 8001 |
| Compose doesn't accept `required` under `env_file` | Compose is older than 2.24. Update Docker |
| Replies say a colleague will follow up, with `model unavailable` | Check that `LLM_BASE_URL` uses `host.docker.internal` and that Ollama is running. On Linux, start it with `OLLAMA_HOST=0.0.0.0` |
| The build fails while downloading | It needs internet access to PyPI, download.pytorch.org and huggingface.co. Behind a proxy, set the proxy in Docker's settings |

## Getting started

To run it without Docker, follow these steps. You need Python 3.12 or later (tested on 3.14) and Git. A model is optional: without one, the app runs on a built-in stand-in, so everything works offline.

The commands are for Windows Command Prompt. Where PowerShell or macOS/Linux differ, it's noted under the step.

### 1. Get the code

```
git clone https://github.com/AKSHAY526/TrustDesk-AI-Support-Operations-Agent.git
cd TrustDesk-AI-Support-Operations-Agent
```

### 2. Create a virtual environment and install

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The prompt starts with `(.venv)` once it's active. Every new terminal needs the activate line again, but the install only happens once. It takes a few minutes, because `sentence-transformers` pulls in PyTorch.

- PowerShell: activate with `.\.venv\Scripts\Activate.ps1`. If it says running scripts is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again. That only lasts for the current window.
- macOS/Linux: `python3 -m venv .venv`, then `source .venv/bin/activate`.

### 3. Connect a model (optional)

Skip this to use the stand-in. To use Llama 3.2 through [Ollama](https://ollama.com):

1. Install Ollama and run `ollama pull llama3.2`.
2. Check it's running: `curl http://localhost:11434` should print `Ollama is running`.
3. Copy the settings file: `copy .env.example .env` (PowerShell: `Copy-Item .env.example .env`, macOS/Linux: `cp .env.example .env`).
4. Open `.env` and replace the four `LLM_` lines at the top with:

```ini
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2
LLM_TIMEOUT_SECONDS=180
```

Ollama ignores the key, but it can't be empty, because an empty key selects the stand-in. The long timeout is for CPU inference, where triaging and drafting one ticket takes around half a minute. Other providers are under [Choosing a model](#choosing-a-model).

### 4. Load the seed data

```
python -m scripts.seed --reset
```

This creates `trustdesk.db` from the files in `data/` and indexes the policies for search. The first run also downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB). It should finish with:

```
  customers      6
  orders         6
  tickets        8
  tools          6
  eval_cases     8
  kb_documents   8
  kb_chunks      25
  embedder       sentence-transformers/all-MiniLM-L6-v2
```

### 5. Start the server

```
python -m uvicorn app.main:app --reload
```

Wait for `Application startup complete`. The search model loads before that line appears, so it can take a little while. Keep this terminal open; Ctrl+C stops the server. `--reload` restarts it when the code changes but not when `.env` does, so restart it yourself after changing settings.

### 6. Open the app

- http://127.0.0.1:8000/ is the console
- http://127.0.0.1:8000/docs is the API reference
- http://127.0.0.1:8000/health shows which model and embedder are active

The console starts you as a support agent, and **Signed in as** at the top switches between agent, manager and admin. Each role sees different panels, as listed under [Roles](#roles). Click a ticket in the queue, then **Run triage** and **Generate reply**. The [demo walkthrough](#demo-walkthrough) is a full tour.

The `/docs` page lists every endpoint and schema, but its "Try it out" button can't send the `Authorization` header, so calls from there come back `401`. Use the console, or send the header yourself. This works in both Command Prompt and PowerShell:

```
curl.exe -X POST http://127.0.0.1:8000/tickets/tkt_9001/draft -H "Authorization: Bearer demo-agent-token"
```

### 7. Run the tests and evals

Open a second terminal in the project folder so the server keeps running, and activate the environment there too:

```
.venv\Scripts\activate
pytest
python -m scripts.run_evals
```

The tests always use the stand-in, so they pass with or without Ollama. The evals use whichever model `.env` selects, which takes a few minutes with llama3.2 on a CPU.

### 8. Start again from clean data

Stop the server with Ctrl+C, then:

```
python -m scripts.seed --reset
python -m uvicorn app.main:app --reload
```

This clears every triage, reply and action recorded since the last seed. Stop the server first, because Windows won't delete a database file that's in use.

### Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again |
| `'ls' is not recognized` | Command Prompt uses `dir` |
| `No module named 'fastapi'` (or similar) | The environment isn't active in this terminal. Run `.venv\Scripts\activate` |
| `[Errno 10048]` or "address already in use" | A server is already running on port 8000. Stop it, or start with `--port 8001` and use that port in the URLs |
| `PermissionError` while seeding | The server is still running. Stop it and seed again |
| The reply says a colleague will follow up, and the escalation reason says `model unavailable` | Ollama isn't running, the model isn't pulled, or the call timed out. Check `curl http://localhost:11434` and `ollama list`, or raise `LLM_TIMEOUT_SECONDS` |
| `/health` still shows `heuristic-stand-in` after editing `.env` | Restart the server, and check that `LLM_API_KEY` isn't empty |
| `StaleEmbeddingsError` in the server log | The database was indexed with a different embedder. Stop the server and run `python -m scripts.seed --reset` |
| The queue stays on "Loading…" | Refresh with Ctrl+F5. If it's still stuck, the server terminal shows the error |

## Configuration

Settings come from environment variables, or from `.env` if it exists. `.env` is gitignored, and none of these values are ever put into a prompt.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | empty | OpenAI-compatible base URL |
| `LLM_API_KEY` | empty | Leave empty to use the offline stand-in |
| `LLM_MODEL` | `gemini-2.0-flash` | Model name |
| `LLM_TIMEOUT_SECONDS` | `20` | Timeout for each model call |
| `TOKEN_AGENT` | `demo-agent-token` | Token for the agent role |
| `TOKEN_MANAGER` | `demo-manager-token` | Token for the manager role |
| `TOKEN_ADMIN` | `demo-admin-token` | Token for the admin role |
| `RETRIEVAL_MIN_SCORE` | `0.35` | Below this, the agent asks or escalates instead of answering |
| `RETRIEVAL_TOP_K` | `4` | Number of policy sections passed to the model |

### Choosing a model

Any OpenAI-compatible chat completions endpoint works, since `app/llm.py` is the only file that talks to a provider. To switch, change the `LLM_` lines in `.env` and restart the server. `/health` and the console header show the model in use.

| Provider | `LLM_BASE_URL` | `LLM_API_KEY` | `LLM_MODEL` |
|---|---|---|---|
| Ollama, on your machine | `http://localhost:11434/v1` | `ollama` | `llama3.2`, or any name from `ollama list` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | your Gemini API key | `gemini-2.0-flash` |
| OpenAI | `https://api.openai.com/v1` | your OpenAI API key | `gpt-4o-mini` |

For another Ollama model, pull it first (for example `ollama pull qwen2.5:3b`) and use the same name in `LLM_MODEL`. Hosted providers answer in a few seconds, so the default 20 second timeout is enough for them. An empty `LLM_API_KEY` switches back to the stand-in.

Keep comments on their own lines in `.env`. A `#` after a value is read as part of the value.

Ollama is a separate application rather than a Python package, so nothing about it is in this repo.

## Project layout

```
app/
  main.py         FastAPI app
  routers/        HTTP endpoints
  agent.py        triage, draft replies, action recommendations
  retrieval.py    hybrid search and the coverage threshold
  guardrails.py   injection screening and reply checks
  tools.py        tool catalogue, approvals, idempotency
  llm.py          model adapter: real provider, stand-in, test doubles
  embeddings.py   sentence-transformers, with a hashing fallback
  loader.py       seed loading and chunking
  db.py           schema and connections
  repository.py   read queries
  traces.py       run logging
  evals.py        eval runner
data/             seed pack from Airtribe, unchanged
scripts/          seed.py, run_evals.py
static/           console (a single HTML file)
tests/
Dockerfile        image with CPU-only PyTorch, the embedding model and seeded data
docker-compose.yml
requirements.txt  direct dependencies
constraints.txt   exact version of every installed package, used by the Docker build
```

## How a ticket is handled

```
console --REST--> FastAPI (auth, roles, validation)
                    |-- retrieval    FTS5 + embeddings over policy sections
                    |-- agent        triage and drafts, through the LLM adapter
                    |-- guardrails   screening, poisoned documents, reply checks
                    `-- tools        catalogue rules, approval, idempotency
                  everything stored in SQLite (one file)
```

1. Load the ticket with its customer and order. Nothing about other customers is loaded.
2. Screen the ticket text for injection attempts.
3. Search the policies. Sections that contain instructions are dropped.
4. Triage. A label outside the allowed set is rejected and retried once; if it still fails, the ticket goes to a human.
5. Draft a reply from the order details and policy sections. Citations that weren't in the search results are removed, and a reply with no valid citation isn't presented as a policy answer.
6. Recommend a tool based on the order and the policy: return window, final sale, tracking, duplicate charges, safety.
7. Low-risk tools run straight away. The rest wait for a manager.
8. Write a trace for the run.

## Roles

The brief asks for simple authentication, either a demo token or a login flow. This uses demo tokens: one per role, set in `.env` and checked by the server on every request. The console's **Signed in as** menu picks which token it sends.

| | Support agent | Support manager | Admin |
|---|---|---|---|
| Ticket queue, ticket details, policy search | ✓ | ✓ | ✓ |
| Run triage, generate a reply, request an action | ✓ | ✓ | ✓ |
| See the status of a ticket's actions | ✓ | ✓ | ✓ |
| Approve or reject an action | | ✓ | ✓ |
| Waiting for approval, across all tickets | | ✓ | ✓ |
| Policy gaps | | ✓ | ✓ |
| Run evaluations and see the results | | | ✓ |
| Add policy documents (`POST /kb/ingest`, API only) | | | ✓ |

The console hides what a role can't use, and the API refuses it with `403 insufficient_role`, so hiding a panel isn't the only protection. Nobody can approve an action they requested themselves, whatever their role.

An action that needs approval shows "Sent for manager approval" until a manager decides. Low-risk tools (`open_carrier_investigation` and `escalate_to_human`) don't need approval, so they run as soon as they're requested and show "Approval not required".

The menu is a demo shortcut, so anyone using the console can pick any role. A real deployment would have each person sign in and get only their own role.

## API

Every API endpoint except `/health` needs `Authorization: Bearer <token>`. Request and response schemas are at `/docs`.

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/tickets` | any | Ticket list |
| GET | `/tickets/{id}` | any | Ticket with its customer, order, triage, drafts and actions |
| POST | `/tickets/{id}/triage` | any | Triage a ticket |
| POST | `/tickets/{id}/draft` | any | Draft a cited reply and recommend actions |
| GET | `/kb/search?q=` | any | Policy search, with a `covered` flag |
| POST | `/kb/ingest` | admin | Add or replace policy documents |
| GET | `/actions/catalogue` | any | The tool catalogue |
| POST | `/actions` | any | Request an action |
| GET | `/actions?ticket_id=` | any | One ticket's actions |
| GET | `/actions` | manager, admin | Actions across all tickets. `?status=approval_required` gives the approval queue |
| GET | `/actions/{id}` | any | Fetch one action |
| POST | `/actions/{id}/approve` | manager, admin | Approve or reject, then execute |
| GET | `/traces/{run_id}` | any | Trace for one AI run |
| GET | `/tickets/{id}/traces` | any | All traces for a ticket |
| GET | `/insights/policy-gaps` | manager, admin | Escalation reasons |
| POST | `/evals/run` | admin | Run the evals (`?background=true` to queue the run) |
| GET | `/evals/latest` | admin | Latest eval report |
| GET | `/health` | none | Active model and embedder |

Errors look like `{"detail": {"error": "amount_over_limit", "message": "..."}}`. Other codes include `invalid_token`, `insufficient_role`, `unknown_tool`, `missing_required_fields`, `tool_not_allowed_for_category`, `separation_of_duties` and `ticket_not_found`.

Approving a replacement:

```bash
AGENT="Authorization: Bearer demo-agent-token"
MANAGER="Authorization: Bearer demo-manager-token"

curl -s -X POST localhost:8000/tickets/tkt_9001/draft -H "$AGENT"

curl -s -X POST localhost:8000/actions -H "$AGENT" -H "Content-Type: application/json" -d '{
  "ticket_id": "tkt_9001",
  "tool_name": "create_replacement_order",
  "payload": {"order_id": "ord_5001", "sku": "BG-AIRPODS-01",
              "reason": "Damaged on arrival", "idempotency_key": "tkt_9001-replacement-1"}}'
# status: approval_required

curl -s -X POST localhost:8000/actions/<action_id>/approve -H "$AGENT" \
  -H "Content-Type: application/json" -d '{"decision": "approved"}'
# 403 insufficient_role: agents can't approve, even without the console

curl -s -X POST localhost:8000/actions/<action_id>/approve -H "$MANAGER" \
  -H "Content-Type: application/json" -d '{"decision": "approved"}'
# status: executed
```

## Design notes

**SQLite.** One file and nothing to install, and the seed pack already ships a SQLite loader. The `UNIQUE` constraint on `idempotency_key` is what makes retries safe. Moving to Postgres would mostly mean changing `db.py` and `repository.py`.

**Chunking and hybrid search.** Policies are split on `##` headings, so search returns the relevant section instead of a whole file. Each query goes through FTS5, which is good at exact identifiers like `KB-REFUND-001`, and through embeddings, which handle paraphrases. The two scores are combined 0.4 / 0.6. BM25 scores are mapped onto a fixed scale instead of being normalised per query; per-query normalisation gives a single weak match a score of 1.0, so off-topic questions looked answerable.

**Coverage threshold.** If the best match scores below `RETRIEVAL_MIN_SCORE`, the agent asks for more detail or escalates instead of answering. Coverage is checked against the ticket text only. The search used for citations also includes order details, which helps ranking but was making off-topic questions look covered.

**Eligibility is computed, not generated.** Return windows are checked in code against the ticket's `created_at`, not today's date, so the seed data doesn't go stale. Final-sale items, stale tracking, duplicate charges and safety wording are also read from the ticket and order rather than left to the model.

**One model adapter.** Only `app/llm.py` calls a provider. Tests swap in the stand-in, a scripted model, or one that always fails.

**Not blocking.** Endpoints are plain `def` functions, so FastAPI runs them in a threadpool and a slow model call doesn't hold up other requests. The eval run can also be queued with `?background=true`.

**Escalation reasons are kept.** `/insights/policy-gaps` lists why tickets were sent to a human, which doubles as a list of missing policies.

## Guardrails

Pattern matching will eventually miss something, so there are several layers:

1. Ticket and policy text go into prompts as delimited data blocks, separate from the instructions.
2. Incoming messages are checked against known patterns, then by the model to catch rewordings. If the patterns already flag a message, the model check is skipped.
3. Search results that contain instructions are dropped. That's how `KB-ADVERSARIAL-001` is handled, without special-casing it, and it doesn't escalate the customer's ticket.
4. Replies are checked for configured secrets and for citations that weren't retrieved.
5. The tool engine's limits: allowed categories, amount caps and approval. These hold even when everything above misses.

Secrets are never put into prompts. The console escapes everything it renders, since some seed tickets contain injection text.

## Evaluation

```bash
python -m scripts.run_evals                  # table
python -m scripts.run_evals --json
python -m scripts.run_evals --case eval_006
```

This runs `data/eval_cases.jsonl` through the same code the API uses, and exits non-zero if any case fails. The `expected_*` labels are only read here.

| Metric | Stand-in | llama3.2 (3B, Ollama) |
|---|---|---|
| Triage accuracy (category and priority) | 100% | 75% |
| Category accuracy | 100% | 87.5% |
| Priority accuracy | 100% | 75% |
| Citation coverage | 100% | 100% |
| Unsafe action block rate | 100% | 100% |
| Escalation accuracy | 100% | 100% |
| Allowed-action accuracy | 100% | 100% |
| Cases fully passed | 100% | 75% |

The stand-in is a keyword classifier written against these eight tickets, so its 100% mainly shows that the pipeline works end to end. The llama3.2 run is the realistic one. It gets two triage calls wrong (a stuck parcel rated `medium` instead of `high`, and the coupon injection filed under `account_security` instead of `general`), but none of the safety metrics drop, because those are decided in code.

With sentence-transformers, the real tickets score 0.40 or higher and unrelated questions 0.30 or lower, so the 0.35 threshold sits between them.

## Tests

`pytest` runs 125 tests in under half a minute. They always use the stand-in model and the hashing embedder, so they run offline and give the same result every time, even when `.env` points at a real provider.

Besides the normal flows, they cover:

- retrieval ranking, the coverage threshold, and FTS operators in ticket text
- each guardrail layer, including the poisoned policy document
- invalid triage labels, provider failures, and return windows measured from the ticket date
- tool rules, separation of duties and idempotency
- roles, including which roles the API lets see the approval queue, policy gaps and eval results
- ten threads approving the same action at once, which must produce exactly one replacement order
- a model that has been talked into requesting a ₹5000 coupon, which the ₹1000 cap refuses
- fifteen reworded attacks, none of which may lead to a money-moving action

## Demo walkthrough

In the console:

1. As the support agent, open `tkt_9001` (damaged earbuds), run triage and generate a reply. It should come back as `refund`, cite `KB-REFUND-001`, and suggest `create_replacement_order`. Request it, and it shows "Sent for manager approval" with no Approve button. Switch to support manager: the request is under **Waiting for approval**. Open it and approve it, and it executes. The `curl` example under [API](#api) shows the server refusing an agent's approval too.
2. Request the same action again. Nothing new is created.
3. `tkt_9006` uses a prompt injection to ask for a ₹5000 coupon. The reply refuses, cites `KB-SECURITY-001`, and only an escalation is suggested.
4. `tkt_9007` asks for the system prompt and API key. Refused and escalated.
5. `tkt_9003` is a final-sale software licence. The refund is declined, with no action and no escalation.
6. `tkt_9004` is a swollen battery outside its warranty. It's marked `urgent` and escalated.
7. Switch to admin and run the evaluation. Only the admin sees this panel.
8. As a manager or admin, check the policy gaps panel for the escalation reasons.

## Known limitations

- Single node. SQLite serialises writes. Idempotency and the approval state machine already live in the database, so scaling out would mean Postgres, a queue for the AI work, and several API instances.
- Small eval set. There are eight cases and the triage prompt was adjusted against them, so treat the scores as a regression check rather than a measure of general accuracy.
- Triage quality depends on the model, as the llama3.2 results show.
- The fallback embedder is lexical. Without sentence-transformers, real tickets and off-topic questions score about the same (roughly 0.40 against 0.41), so the coverage threshold isn't reliable in that mode. `/health` shows which embedder is active, and switching embedders without re-seeding raises an error.
- Screening can be evaded by new phrasings. The tool limits are what stop that from causing harm.
- Customer names and order details are sent to the model provider. That's fine for this fictional data, but production use would need redaction.
- No multi-turn conversation. If details are missing, the reply asks for them and the ticket is marked `needs_info`.
- Tools are mocked, so nothing external is called.
- No API rate limiting, and traces are never cleaned up.

## Credits

Problem statement and seed data: [airtribe-projects/trustdesk-capstone](https://github.com/airtribe-projects/trustdesk-capstone).

Dependencies: FastAPI (MIT), Uvicorn (BSD), httpx (BSD), NumPy (BSD), sentence-transformers (Apache 2.0), PyTorch (BSD) and pytest (MIT), pinned in `requirements.txt`, with every installed version frozen in `constraints.txt`.

Built with AI assistance, which the capstone allows.
