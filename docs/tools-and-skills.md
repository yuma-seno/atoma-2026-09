# Tools and Skills

Atoma runtime exposes two tool categories:

- External MCP tools loaded from `tools.yaml` and discovered from running servers.
- One built-in Atoma tool: `atoma_builtin__load_skill`.

The built-in skill loader is always present and is not configured in `tools.yaml`.

## External MCP tools

`tools.yaml` maps server names to process definitions.

Minimal shape:

```yaml
filesystem:
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
  env: {}
  hooks:
    tool_allowlist: ["filesystem__read_file"]
    before_tool: ./scripts/before_hook
    after_tool: ./scripts/after_hook
```

At runtime:

- Atoma starts each configured server, or connects to one that is already running.
- Calls MCP `tools/list` and registers each tool as `server__tool`.
- Converts MCP `inputSchema` to OpenAI-compatible `function.parameters`.

### Where a server is

`command` starts one. `url` reaches one that is already running, over MCP's
Streamable HTTP transport. Both together start one and then speak HTTP to it.

```yaml
# A child process, over stdio. The usual case.
shell:
  command: bun
  args: ["run", "./scripts/shell.ts"]

# Already running somewhere else.
warehouse:
  url: https://mcp.internal.example.com/mcp
  headers:
    Authorization: "Bearer ${WAREHOUSE_TOKEN}"

# Started here, spoken to over HTTP.
indexer:
  command: ./scripts/indexer
  url: http://127.0.0.1:9137/mcp
  env:
    GH_TOKEN: "${GH_TOKEN}"
```

A server with neither is refused when the tools file is read, as is a `url` that is
not `http://` or `https://`.

**A credential reaches a process through `env` and an endpoint through `headers`,
and neither substitutes for the other.** `env` is applied to a child's environment,
so on a server with a `url` and no `command` there is nothing to apply it to —
that combination is refused rather than ignored, because a credential you believe
you routed and did not is worse than an error. The mirror is refused for the same
reason: `headers` on a server with no `url` would be a token that is never sent.
Both lists are still per server, so a value reaches one server only by being named
in that server's own block.

That is the routing mechanism in both directions. A server atoma starts inherits
this process's environment with the credential names removed first and its own
`env` applied second — so naming `GH_TOKEN: "${GH_TOKEN}"` here is what puts it
back, for that server alone, and a server that names nothing gets nothing. Which
names are removed is the provider credentials plus whatever `protect_env` declares
in `atoma.toml`; see [configuration.md](configuration.md).

What the third form buys is worth stating, because it looks redundant. A server
atoma starts is one whose **stderr and stdout atoma owns**, so its reports still
reach the agent (see below) — and one that can be handed a credential through its
environment, which is how most servers are written to take one. A server somebody
else is running has neither: it is authenticated by what it already holds, and
whatever it logs, it logs where atoma cannot see.

| | `command` | `url` | both |
|---|---|---|---|
| transport | stdio | Streamable HTTP | Streamable HTTP |
| credential | `env` | `headers` | `env`, and `headers` if it wants them |
| its logs | stderr | not visible here | stderr and stdout |
| lifetime | the run | outlives every run | the run |
| startup cost | every run | paid once, elsewhere | every run |

A server atoma starts and then reaches over HTTP is not listening the instant it is
spawned, so `initialize` is retried until it answers, bounded by
`ATOMA_MCP_INIT_TIMEOUT` (120s). A `url` atoma did **not** start is tried once: it is
either up or the address is wrong, and retrying a wrong address for two minutes
turns a typo into a hang.

### What Streamable HTTP does and does not do here

- One POST per message. The response is either one JSON object or an SSE stream,
  and both are accepted — the server chooses.
- A `Mcp-Session-Id` the server assigns at `initialize` is sent back on every
  later request, and so is the protocol version the server answered with.
- **Not implemented:** the older HTTP+SSE transport (two endpoints, deprecated in
  the specification since 2025-03-26), the optional `GET` stream for
  server-initiated messages, resumption with `Last-Event-ID`, and OAuth. A token
  goes in `headers`.
- A session is not deleted when the run ends. Servers time them out; atoma does not
  send the optional `DELETE`.

### How long a server has to answer

`request_timeout_secs` sets the limit for one `tools/list` or `tools/call` on that
server. Omit it and the client's default applies (60s, or `ATOMA_MCP_TIMEOUT`).

```yaml
shell:
  command: bun
  args: ["run", "./scripts/shell.ts"]
  request_timeout_secs: 3600
```

Set it when the server's work genuinely takes minutes — a shell server running a
build or a test suite, or one that loads a model before it can answer. **A tool's
own timeout argument does not raise this limit.** A `shell_execute` that accepts
`timeout_seconds: 600` still gets cut off at 60 unless the server says so here,
and the error names the tool rather than the limit that caused it.

Leave it alone otherwise. This is the only thing that notices a server which has
stopped responding, so a large value is a long wait before a stuck run says so.
Setting it on a server that answers in milliseconds trades away the detection and
buys nothing.

`0` means the default, the same as omitting it.

### When a server is not well

A server can degrade without failing. The one this project ships that loads a
reranker kept answering every search after the model failed to load -- with
first-stage results, which look exactly like results. Nothing noticed for two
releases.

So a server's report about itself travels with its next tool result, named as
coming from the server:

```
...the tool's answer...

--- 1 problem reported by the 'search' server, not part of the answer above ---
warning: could not preload the reranker (EACCES), results are first-stage ordered
```

Nothing is attached on a healthy call, and there is nowhere for an agent to go and
look: it arrives where it is used. Warnings and errors only; anything a server logs
at `info` or below stays in the run log.

Two channels feed it, whatever transport the server speaks:

- **`notifications/message`**, MCP's `logging` capability. Atoma declares it,
  asks for `warning` and above with `logging/setLevel`, and takes the severity
  from the notification's `level`. This is the one to implement: it works over any
  transport and the server says how bad the thing is.
- **the server's own output**, for a server that implements no logging capability --
  which today is every third-party one. **Off unless that server asks for it**,
  with `guess_severity_from_output`:

  ```yaml
  search:
    command: bun
    args: ["run", "./scripts/search.ts"]
    guess_severity_from_output: true
  ```

  Because there is no severity field here, one has to be read out of the words
  (`error`, `errors`, `fatal`, `panic`, `warn`, `warning`, `warnings`) — a guess,
  and one calibrated on servers whoever wrote the list had read. Atoma ships no MCP
  server, so that calibration does not travel. `npx -y
  @modelcontextprotocol/server-filesystem` prints npm's deprecation notice before
  the server starts, and every first tool result carried "1 problem reported by the
  'filesystem' server" describing npm; a build tool that ends with "0 errors" is the
  same mistake pointed the other way. Whoever connected the server could not correct
  either, because the output is not theirs. So it is per server, set by someone who
  has read that server's output — the same people who set its timeout.

  Off is not silent. Every line still goes to the run log, exactly as before; what
  stops is the guess being attached to a tool result as the server's own report.

  This channel only exists for a server atoma started. Over stdio that means
  stderr, because stdout is the transport; over HTTP it means stderr **and**
  stdout, since neither is. **A server atoma did not start has no such channel at
  all** — it logs where atoma cannot read — so for a remote server,
  `notifications/message` is the only way a problem reaches the agent.

The same report is attached once, not on every call after it, and at most twenty
distinct ones travel with a single result -- past that a count goes in their place.

For a server this project owns, log to `notifications/message`. A message an agent
cannot act on costs it a result it has to read past, so what belongs at `warning`
is what changed about the answer, not that something happened.

## What atoma finds wrong with a tools file

The section above is what a server says about itself. This one is atoma's own audit
of your configuration, and it goes to the caller rather than to the agent: an agent
mid-run cannot fix the file it was started with.

Both `atoma run` and `atoma validate --with-live-tools` start every declared server,
ask it `tools/list`, and compare the answer with the file. Two things are checked:

| Finding | Severity | What it means |
| --- | --- | --- |
| `dead_guard` | `warn` | A pattern in `tool_allowlist` or `tool_denylist` matches none of the tools that server advertises. It reads as a guard and guards nothing — usually a `server__` prefix left in place on a server that sets `unprefixed: true`. |
| `duplicate_tool` | `error` | Two servers offer one tool name, which routing cannot resolve. The run stops. |

**Only `atoma run` writes this line.** `atoma validate --with-live-tools` prints the
same findings as prose on stderr and exits non-zero: it is a gate, not a channel.

Each finding is written once, to the log stream on stderr, as a line of fields:

```
ATOMA_CONFIG_FINDING: kind=dead_guard severity=warn server=files_ro pattern=files_ro__* tools=read,grep,glob
ATOMA_CONFIG_FINDING: kind=duplicate_tool severity=error tool=read servers=files,files_ro
```

`kind=dead_guard` carries `server`, `pattern` and `tools` (every tool that server
advertises, comma-separated, empty when it advertises none). `kind=duplicate_tool`
carries `tool` and `servers`. Values percent-encode whitespace, `,` and `%`, and are
otherwise literal — so `a server` with a space in its name arrives as
`server=a%20server` rather than as two fields.

Three things are worth knowing about this line:

- **The fields are the contract; the English sentence beside them is not.** The prose
  is written for a person and may be reworded in any release. Anything acting on a
  finding should read `kind=` and the fields next to it.
- **An `error` finding emits its line and then stops the run**, before any `--output
  json` envelope exists. That is the case the line exists for: a fatal finding is
  precisely the one a caller has no other channel to hear about.
- **`severity` is atoma's verdict about the configuration, not your policy about it.**
  `--fail-on-tool-findings` changes what a run does with a `severity=warn` line; it
  does not change what the line says.

The level follows the severity: `severity=error` is logged at `error` and
`severity=warn` at `warn`, so a line survives any filter that would still show the
prose it belongs to. A filter above `error` hides both.

`atoma run --fail-on-tool-findings` refuses to run when there is any finding at all.
It is off by default, and deliberately: stopping the run does not close a guard that
has stopped guarding. It only removes the ability for an agent to repair the
configuration, leaving a person to do it by hand. The default is to say so — in the
line above, which the environment around atoma can act on — and go on.

For a pull-request gate, `atoma validate --with-live-tools` is the better place. It is
already strict about both findings, it starts the servers without running an agent,
and it fails before anything has been changed.

## Prefixes and reserved namespaces

- External tool names are always prefixed by server name: `server__name`.
- Prefix `atoma_builtin__` is reserved.
- If any external tool starts with `atoma_builtin__`, runtime fails during startup.

## Hooks and failure behavior

Hook order:

1. denylist/allowlist check
2. `before_tool` hook
3. MCP tool call
4. `after_tool` hook

Hook rules:

- `tool_allowlist` and `tool_denylist` may both be set. The denylist is checked first,
  so a tool matching both is blocked. Setting both logs a warning naming that order,
  because it is unusual rather than wrong — it used to be refused as "ambiguous" while the
  code, this document and a test all described the precedence. `atoma validate` says the
  same thing at the same volume and also goes on; `atoma validate --strict` is how a
  caller whose own policy forbids it gets a non-zero exit. See
  [configuration.md](configuration.md#validation-workflow).
- Pattern matching supports exact match or trailing `*` wildcard.
- `before_tool` is fail-closed:
  - non-zero exit, timeout, or invalid JSON blocks the call.
  - expected output JSON includes `allow: true`.
- `after_tool` is best effort:
  - failures are logged and do not fail the tool call.

Related env timeouts:

- `ATOMA_HOOK_TIMEOUT` (default 30s)
- `ATOMA_MCP_TIMEOUT` (default 60s) — the default for every server; a server's own
  `request_timeout_secs` wins over it
- `ATOMA_MCP_INIT_TIMEOUT` (default 120s)

Every one of these treats `0`, blank, and unparseable as "use the default".

## When a call times out

The call fails and the agent sees an error naming the tool. The server is not
told: it keeps working and eventually writes its answer, which arrives after
nobody is waiting for it.

Atoma reads past that answer and discards it, matching the JSON-RPC `id` to the
request in flight. It logs a `warn` naming both ids when it does, which is the
signal that a timeout fired on that server earlier in the run.

This matters because the alternative is silent: without the id check, the next
call on that server reads the abandoned answer, and every answer from then on
belongs to the previous question for the rest of the run. Nothing about the shape
of the result gives it away.

## Skill catalog format

Skills are loaded recursively from `--skills-dir`.

Each `.md` file must include:

- YAML frontmatter with non-empty `name` and `description`
- non-empty instruction body

Example:

```markdown
---
name: engineering/tdd
description: Test first.
---

Write a failing test before implementation.
```

Loader safety checks:

- symbolic links are rejected
- duplicate `name` values are rejected
- malformed frontmatter or empty instructions fail startup

## Progressive disclosure of skills

At prompt build time, Atoma reveals only skill metadata in `AVAILABLE_SKILLS`.

When an agent needs details, it calls:

- `atoma_builtin__load_skill` with `{"skill_name":"..."}`

Tool result returns full skill instructions as normal tool output, which is persisted in session history like any other tool call.

## Built-in loader is not configurable via `tools.yaml`

`tools.yaml` can only define external MCP servers.

You cannot:

- remove `atoma_builtin__load_skill`
- rename it
- add custom parameters to it

Those behaviors are implemented directly in runtime code.
