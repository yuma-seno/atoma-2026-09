# Configuration

Atoma resolves runtime settings from CLI flags, `atoma.toml`, and hard defaults.

## Generate `atoma.toml`

```bash
atoma init > atoma.toml
```

If you are running from source:

```bash
cargo run -- init > atoma.toml
```

Atoma discovers `atoma.toml` from the current directory upward through ancestor directories.

## Precedence rules

Run settings precedence (high to low):

1. CLI argument
2. selected profile in `atoma.toml`
3. `[defaults]` in `atoma.toml`
4. hard-coded default

Environment variable values loaded from config follow:

1. existing process environment
2. `[profile.<name>.env]`
3. top-level `[env]`

At runtime, config-defined env values are only injected when that key is not already set in the process.

## What a tool server does not inherit

A tool server atoma starts inherits this process's environment **minus the names atoma
treats as credentials**. It is not `env_clear()`: a server needs `PATH` to find its
interpreter and `HOME` for its caches, and an allowlist of what a runtime needs is
wrong the first time somebody adds a server written in another language.

Two things make up that list, and neither is written out by hand:

1. **Every provider credential this build knows** — including the names a provider
   only falls back to, such as `GITHUB_TOKEN` and `GH_TOKEN` for `github-copilot`.
   Adding a provider covers its names automatically, and the list is the whole
   provider table rather than the provider this run resolved to, so a name is
   stripped whichever provider is in use.
2. **Whatever `protect_env` names.** atoma cannot know your secrets — GitLab's
   `CI_JOB_TOKEN`, a Slack bot token, an in-house credential named after an in-house
   system — so it takes them as values instead of guessing:

```toml
protect_env = ["CI_JOB_TOKEN", "SLACK_BOT_TOKEN"]
```

The key is **top-level**, above any `[table]` header. A bare key written below one
belongs to that table and would do nothing. It is deliberately not per-profile: a
name protected under one profile and inherited under another is the kind of
difference nobody notices until it has leaked, so this is a floor for the whole
project rather than a run setting.

The removal happens before a server's own `env` is applied, so a server that
legitimately needs one of them names it in its own `env` in the tools file and gets
it back — for that one server and no other. See
[tools-and-skills.md](tools-and-skills.md).

In `--credentials-file` mode this is mostly moot, because none of these are in
atoma's environment to begin with. It earns its place when atoma is run by hand with
the secrets exported.

## Provider selection

Provider resolution order:

1. `provider:` in agent frontmatter
2. `ATOMA_PROVIDER` environment variable
3. auto-detection

One provider, one credential, one endpoint, and one variable that moves it:

| Provider | Credential | Endpoint | Override |
| --- | --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` | `OPENAI_BASE_URL` |
| `openai-responses` | `OPENAI_API_KEY` | `https://api.openai.com/v1` | `OPENAI_BASE_URL` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | `OPENROUTER_BASE_URL` |
| `openrouter-responses` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | `OPENROUTER_BASE_URL` |
| `orcarouter` | `ORCAROUTER_API_KEY` | `https://api.orcarouter.ai/v1` | `ORCAROUTER_BASE_URL` |
| `orcarouter-responses` | `ORCAROUTER_API_KEY` | `https://api.orcarouter.ai/v1` | `ORCAROUTER_BASE_URL` |
| `anthropic` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` | `ANTHROPIC_BASE_URL` |
| `github-copilot` | `ATOMA_COPILOT_TOKEN` | `https://api.githubcopilot.com` | `COPILOT_BASE_URL` |

`openai` and `openai-responses` are one vendor reached two ways — chat completions
and the Responses API — so they share a credential and an endpoint. That pair is also
how to reach a provider with no row of its own: point `OPENAI_BASE_URL` at anything
that speaks either dialect. What you give up by doing that instead of using a named
provider is that the run's log says `openai`, so where it went is only visible in the
environment.

Each router serves both dialects, so each combination has a name. That is what puts
the destination in a run's log: `openai-responses` with `OPENAI_BASE_URL` pointed at a
router logs `openai-responses`, and only the environment says where it went.

**Auto-detection is by credential.** Exactly one present selects that provider; two
present is an error naming both, because which to use is not something the
credentials decide — name it with `ATOMA_PROVIDER` or the agent's `provider:`, or
remove the one this run should not use. When a credential serves more than one row — a
vendor reached by two dialects — the chat-completions one wins, and the Responses one
has to be asked for by name.

`github-copilot` also accepts `GITHUB_TOKEN` or `GH_TOKEN`, and those deliberately
take no part in auto-detection: a run that talks to GitHub has one anyway, so
detecting Copilot from them would make every such run ambiguous. Select
`github-copilot` explicitly to use them.

Routers that attribute requests to an application receive Atoma's own name and
repository (OpenRouter reads them as `X-Title` and `HTTP-Referer`). They are fixed:
the question they answer is which application is asking, and that is this one
whoever deploys it. Who is paying is the API key. Only the providers that read them
receive them.

## Headers an agent adds

An agent definition's `extra_headers` are sent with every request that agent makes,
for a setting a provider reads from a header rather than from the body.

```yaml
extra_headers:
  X-OrcaRouter-Session-Id: atomaton-engineer
```

A header Atoma sets itself is refused by `atoma validate` rather than merged, since
those carry authentication, the shape of the body, the API version, and which model
catalogue the account may reach.

## Request timeout and retries

`ATOMA_LLM_TIMEOUT` bounds a single completion request, in seconds (default `300`). Absent, non-numeric, and zero values fall back to the default.

Treat this as a stall detector rather than a generation budget. A provider that has returned nothing for several minutes has usually stopped responding altogether, and raising the ceiling only delays detecting that. Raise it when a model legitimately generates for longer than the default; lower it to abandon a stalling endpoint sooner.

Each request is attempted up to 3 times, with a 1s then 4s pause between attempts. Retried:

- connect, timeout, and body/decode errors
- HTTP 429 and 5xx
- a response body that was cut off mid-payload

Not retried, because the outcome cannot change:

- any other non-success status
- a provider error object returned under HTTP 200
- a body whose JSON structure does not match the expected response shape

Worst-case wall clock for one request is therefore roughly `3 × ATOMA_LLM_TIMEOUT`. Account for that when raising the timeout under a CI job limit.

## Representative config

```toml
protect_env = ["CI_JOB_TOKEN"]

[defaults]
agent_def = "agents/orchestrator.md"
tools_file = "tools.yaml"
skills_dir = "skills"
template = "prompt-template.md"
output = "text"

[profile.review]
agent_def = "agents/reviewer.md"
output = "json"

[env]
ATOMA_PROVIDER = "openrouter"

[profile.review.env]
ATOMA_PROVIDER = "anthropic"
```

## Paths and profile notes

- Path fields in config (`agent_def`, `tools_file`, `skills_dir`, `template`) are consumed as provided.
- `--profile` requires a discovered `atoma.toml`.
- Unknown profile names fail fast.
- Default `agent_def` is `agent.md` when none is supplied.
- Default `output` is `text`.
- There is no default ceiling on a run. See [runtime.md](runtime.md).

## Validation workflow

Validate definitions before run:

```bash
atoma validate --agent-def agents/orchestrator.md --tools-file tools.yaml
```

Validation checks:

- Agent definition parses.
- `knows_about` files exist and are parseable.
- Each `knows_about` target includes `agent` in `callable_by`.
- `callable_by` values are limited to `user` or `agent`.
- `extra_body` does not include reserved keys `model` or `messages`.
- `extra_body.tools`, when present, is an array so it can be merged with the runtime tool definitions. See [agents.md](agents.md).
- `mcp_servers` entries exist in `tools.yaml` when a tools file is given.

Two findings are warnings rather than errors: a server that sets both
`tool_allowlist` and `tool_denylist` — defined, since the denylist is checked first,
and allowed at run time — and `mcp_servers` named with no `--tools-file` to check
them against, which is a check that did not happen. Both are printed and the command
still succeeds, which is the answer `atoma run` gives for the same configuration.

`--strict` makes every warning an error. Whether a pull request may carry one is the
caller's policy, not something atoma can know, so it is asked for rather than
assumed — the counterpart of `atoma run --fail-on-tool-findings`.

`--with-live-tools` adds one more check that a file alone cannot answer: it starts
every declared server, asks what it advertises, and fails on anything wrong with the
configuration — a guard pattern matching nothing, a tool name two servers claim. That
is the strict check, and the right one for a pull-request gate. See
[tools-and-skills.md](tools-and-skills.md#what-atoma-finds-wrong-with-a-tools-file).
