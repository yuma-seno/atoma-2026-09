use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::domain::tool::{Hooks, ToolDef};
use crate::domain::tool_health::{self, HealthLog, Severity};
use crate::domain::tool_output;
use crate::infra::hooks;

/// How long one `tools/list` or `tools/call` may take, for a server that does not
/// say otherwise.
///
/// Sixty seconds is right for a server that answers from memory or from one HTTP
/// call, which is most of them. It is wrong for two kinds that exist:
///
///   - a shell server, whose whole job is running a build or a test suite. Its
///     own `shell_execute` accepts `timeout_seconds` up to 3600 and defaults to
///     300 -- and every value above 60 was a lie, because this constant killed
///     the call first. The error named the tool, so it read as "the shell server
///     is broken" rather than "the client gave up".
///   - a server that loads a model on its first call. A 544MB reranker took 63.9s
///     to load, measured; this gave up at 60.0s and the answer arrived 15s later.
///
/// So the value belongs to the server, not to this file. `request_timeout_secs`
/// in the tools file is how a server says what it needs; this is what applies
/// when it says nothing.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;
/// How long a server has to answer `initialize`, which includes its own startup.
const DEFAULT_INIT_TIMEOUT_SECS: u64 = 120;

/// The default, overridable by `ATOMA_MCP_TIMEOUT` for a whole run.
///
/// Kept as the fallback rather than removed: an operator debugging a slow runner
/// wants one lever, not an edit to every entry in a tools file. A server that
/// declares its own value wins over both.
fn default_request_timeout() -> Duration {
    crate::infra::timeouts::from_env("ATOMA_MCP_TIMEOUT", DEFAULT_REQUEST_TIMEOUT_SECS)
}

fn init_timeout() -> Duration {
    crate::infra::timeouts::from_env("ATOMA_MCP_INIT_TIMEOUT", DEFAULT_INIT_TIMEOUT_SECS)
}

/// Split an MCP tool result's `content` into the text the model reads and the
/// image blocks it should see.
///
/// Images used to be folded into the text, where `serde_json::to_string` turned
/// each into a base64 blob the model could only read as characters — a picture
/// arriving as noise, and an expensive one. They now travel separately, in MCP's
/// own shape, for the LLM adapters to map.
///
/// Text keeps a `[image]` marker where each picture was, so its position in the
/// result stays legible: "the diagram below" means nothing once the diagram has
/// been lifted out.
///
/// Anything that is neither text nor image is still serialised into the text, as
/// before. An unknown block type is not a reason to lose it.
fn split_content(result: &Value) -> (String, Vec<Value>) {
    let Some(items) = result.get("content").and_then(|c| c.as_array()) else {
        return (
            serde_json::to_string(result).unwrap_or_default(),
            Vec::new(),
        );
    };

    let mut images = Vec::new();
    let parts: Vec<String> = items
        .iter()
        .map(|item| {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                return text.to_string();
            }
            if item.get("type").and_then(Value::as_str) == Some("image") {
                images.push(item.clone());
                return "[image]".to_string();
            }
            serde_json::to_string(item).unwrap_or_default()
        })
        .collect();

    (parts.join("\n"), images)
}

#[derive(Debug, Clone)]
pub struct RegisteredTool {
    /// The name the model calls it by: `server__tool`, or `tool` when the server
    /// sets `unprefixed`.
    pub prefixed_name: String,
    /// The name the server itself knows it by, which is what goes back on the wire.
    /// Derived from the other one by splitting, until a name had nothing to split.
    pub tool_name: String,
    pub schema: Value,
}

/// Where a tool name goes. Built once at registration, because the name no longer
/// carries its own destination.
#[derive(Debug, Clone)]
struct Route {
    server: String,
    tool: String,
}

pub struct McpConnection {
    pub name: String,
    /// Whether this server's tools keep their own names. See `ToolDef::unprefixed`.
    unprefixed: bool,
    /// How this server is reached. Everything else here reads the same for a child
    /// process and for something already running at a url.
    transport: Transport,
    /// The child, when there is one. A server at a url that atoma did not start has
    /// none, and outlives the run.
    process: Option<Child>,
    next_id: u64,
    /// How long this server's `tools/list` and `tools/call` may take. Per server,
    /// because 60 seconds means "stalled" for `github` and "still compiling" for
    /// `shell`.
    request_timeout: Duration,
    /// How much of one result from this server reaches the model. See
    /// `domain::tool_output`: a third-party server has no reason to know about
    /// anyone's context window, so the limit has to be here.
    max_output_chars: usize,
    /// What this server has said about its own trouble, waiting to go out with its
    /// next tool result. See `domain::tool_health` for why a result carries this at
    /// all.
    ///
    /// Behind a lock because two readers fill it: this connection, from
    /// `notifications/message` on stdout, and the detached task reading stderr.
    health: Arc<Mutex<HealthLog>>,
}

impl Drop for McpConnection {
    fn drop(&mut self) {
        if let Some(ref mut process) = self.process {
            match process.try_wait() {
                Ok(Some(_)) => {}
                _ => {
                    let _ = process.start_kill();
                }
            }
        }
    }
}

/// Start the program a definition names, with the environment it declares.
///
/// A tool server gets the credentials its own configuration names, and no others.
///
/// Not `env_clear()`. A tool server needs PATH to find its interpreter, HOME for
/// its package caches, LANG for its encoding, and an allowlist of that would be
/// long, runtime-specific, and wrong the first time someone adds a Python server --
/// exactly the enumeration this codebase has been burned by before. Removing the
/// credentials leaves everything a runtime needs untouched.
///
/// The removal comes first and `envs` second, so a server that declares one of
/// these gets it back. That is the whole routing mechanism: `github` says
/// `GH_TOKEN: ${GH_TOKEN}` and receives it, `shell` says nothing and does not.
///
/// All three pipes, whichever transport follows. What stdout then means differs --
/// see `connect`.
fn spawn_process(config: &ToolDef) -> Result<Child> {
    let mut cmd = Command::new(&config.command);
    cmd.args(&config.args);
    for name in crate::infra::credentials::credential_env_names() {
        cmd.env_remove(name);
    }
    cmd.envs(&config.env);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.spawn()
        .with_context(|| format!("Failed to spawn MCP server: {}", config.name))
}

impl McpConnection {
    /// Reach the server this definition names, starting it first if it names a
    /// command.
    ///
    /// Was `spawn`, and the rename is the point: starting a process is one of the
    /// things this may do rather than the whole of what it is. The three
    /// arrangements are in `domain::tool::ToolDef`.
    pub async fn connect(config: &ToolDef) -> Result<Self> {
        let starts_a_process = !config.command.trim().is_empty();
        let mut process: Option<Child> = None;
        // Held rather than read, so a failure to initialise can put what the server
        // said into the error. The log readers take them once it succeeds.
        let mut stderr = None;
        let mut logged_stdout = None;
        let mut stdio: Option<(ChildStdin, BufReader<ChildStdout>)> = None;

        if starts_a_process {
            let mut child = spawn_process(config)?;
            stderr = child.stderr.take();
            if config.url.is_none() {
                // Over stdio the pipe IS the transport, so nothing else may read it.
                let stdin = child
                    .stdin
                    .take()
                    .context("Failed to capture MCP server stdin")?;
                let stdout = child
                    .stdout
                    .take()
                    .context("Failed to capture MCP server stdout")?;
                stdio = Some((stdin, BufReader::new(stdout)));
            } else {
                // Over HTTP it is not, so whatever the server writes there is
                // ordinary logging and is read like stderr. Left unread it fills the
                // pipe and stalls the server at 64KB -- a hang with no message
                // attached to it.
                logged_stdout = child.stdout.take();
            }
            process = Some(child);
        }

        let transport = match (&config.url, stdio) {
            (Some(url), _) => Transport::Http(Http::new(url.clone(), config.headers.clone())),
            (None, Some((stdin, stdout))) => Transport::Stdio { stdin, stdout },
            // `persistence::tool_def` refuses a server with neither, so a tools file
            // cannot reach this. Named rather than unwrapped, so the two files do not
            // have to be read together to know that.
            (None, None) => anyhow::bail!(
                "MCP server '{}' has neither a command to start nor a url to reach",
                config.name,
            ),
        };

        let mut conn = McpConnection {
            name: config.name.clone(),
            unprefixed: config.unprefixed,
            transport,
            process,
            next_id: 1,
            request_timeout: config
                .request_timeout_secs
                .map(Duration::from_secs)
                .unwrap_or_else(default_request_timeout),
            max_output_chars: tool_output::resolve_limit(config.max_output_chars),
            health: Arc::new(Mutex::new(HealthLog::default())),
        };

        let init_params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": client_capabilities(),
            "clientInfo": {
                "name": env!("CARGO_PKG_NAME"),
                "version": env!("CARGO_PKG_VERSION")
            }
        });
        // A server atoma just started is not listening the instant it is spawned, so
        // the first POST arrives before it binds. Retried only in that case: a url
        // atoma did not start is either up or wrong, and retrying a wrong address for
        // two minutes turns a typo into a hang. Bounded by the same `init_timeout`
        // that already bounds a slow start over stdio.
        let wait_for_it = starts_a_process && config.url.is_some();
        let init =
            tokio::time::timeout(init_timeout(), conn.initialize(init_params, wait_for_it)).await;

        // Whether this server said it can report its own trouble over the protocol,
        // which is what the `logging/setLevel` below is worth sending for. The value
        // of the match rather than a flag set inside it: there is no moment where it
        // holds a guess.
        let server_logs = match init {
            Ok(Ok(response)) => {
                let result = response
                    .get("result")
                    .context("Initialize response missing result")?;
                let server_name = result
                    .pointer("/serverInfo/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let server_version = result
                    .pointer("/serverInfo/version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let protocol = result
                    .get("protocolVersion")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                tracing::info!(
                    "MCP server '{}' connected: {} v{} (protocol: {})",
                    config.name,
                    server_name,
                    server_version,
                    protocol,
                );

                // What the server said it speaks, which its own specification says
                // to put on every later request. atoma asks for 2024-11-05 and a
                // newer server answers with its own version; echoing the answer
                // rather than the question is the difference between agreeing and
                // insisting.
                if let Some(version) = result.get("protocolVersion").and_then(Value::as_str) {
                    conn.transport.negotiated(version);
                }

                let server_logs = result.pointer("/capabilities/logging").is_some();

                if let Some(handle) = stderr {
                    watch_log(
                        config.name.clone(),
                        "stderr",
                        handle,
                        Arc::clone(&conn.health),
                    );
                }
                if let Some(handle) = logged_stdout {
                    watch_log(
                        config.name.clone(),
                        "stdout",
                        handle,
                        Arc::clone(&conn.health),
                    );
                }

                server_logs
            }
            Ok(Err(e)) => {
                let said = what_it_said(stderr, logged_stdout).await;
                anyhow::bail!(
                    "Failed to initialize MCP server '{}': {}{}",
                    config.name,
                    e,
                    said,
                );
            }
            Err(_) => {
                let said = what_it_said(stderr, logged_stdout).await;
                anyhow::bail!(
                    "MCP server '{}' initialization timed out ({}s){}",
                    config.name,
                    init_timeout().as_secs(),
                    said,
                );
            }
        };

        conn.send_notification("notifications/initialized", serde_json::json!({}))
            .await?;

        // Warnings and worse, from a server that said it can send them.
        //
        // Without this the server picks: MCP lets it send at "a default level of
        // its choosing", which in practice is either nothing or every debug line.
        // `warning` is exactly what reaches the agent, so asking for it is asking
        // for what will be used, and nothing else crosses the transport.
        //
        // Best effort. A server that declared the capability and then refuses the
        // request is still a working tool server; failing the connection over its
        // log level would take away more than it protects.
        if server_logs {
            let level = serde_json::json!({ "level": "warning" });
            match tokio::time::timeout(
                conn.request_timeout,
                conn.send_request("logging/setLevel", level),
            )
            .await
            {
                Ok(Ok(_)) => tracing::debug!(
                    "MCP server '{}' will report at warning and above",
                    config.name,
                ),
                Ok(Err(e)) => tracing::debug!(
                    "MCP server '{}' declined logging/setLevel: {}",
                    config.name,
                    e,
                ),
                Err(_) => tracing::warn!(
                    "MCP server '{}' did not answer logging/setLevel",
                    config.name,
                ),
            }
        }

        Ok(conn)
    }

    pub async fn list_tools(&mut self) -> Result<Vec<RegisteredTool>> {
        let response = tokio::time::timeout(
            self.request_timeout,
            self.send_request("tools/list", serde_json::json!({})),
        )
        .await
        .with_context(|| format!("Timed out listing tools from MCP server: {}", self.name))?
        .with_context(|| format!("Failed to list tools from MCP server: {}", self.name))?;

        let tools = response
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
            .context("MCP server did not return tools array")?;

        // A nameless entry fails the listing rather than being registered as `unknown`.
        //
        // MCP requires `name` on a tools/list entry, so this is not a second legitimate
        // shape the way GitHub's seven pull request states are -- it is a server that is
        // not answering the protocol. `unknown` registered it anyway: callable, liable
        // to collide with the next nameless tool from the same server, and indexed under
        // a name no schema in the catalogue matches. Named after the server, because
        // that is what an adopter has to go and fix.
        let registered = tools
            .iter()
            .map(|tool| {
                let tool_name = tool
                    .get("name")
                    .and_then(|n| n.as_str())
                    .with_context(|| {
                        format!(
                            "MCP server '{}' listed a tool with no name, which the protocol \
                             requires. Nothing can call it and nothing else it offers can be \
                             trusted to be described correctly either.",
                            self.name
                        )
                    })?
                    .to_string();
                let prefixed = if self.unprefixed {
                    tool_name.clone()
                } else {
                    format!("{}__{}", self.name, tool_name)
                };
                Ok(RegisteredTool {
                    prefixed_name: prefixed,
                    tool_name,
                    schema: tool.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(registered)
    }

    pub async fn call_tool(
        &mut self,
        tool_name: &str,
        arguments: &Value,
    ) -> Result<(String, Vec<Value>, bool)> {
        let response = tokio::time::timeout(
            self.request_timeout,
            self.send_request(
                "tools/call",
                serde_json::json!({
                    "name": tool_name,
                    "arguments": arguments,
                }),
            ),
        )
        .await
        .with_context(|| {
            format!(
                "Timed out calling tool '{}' on MCP server '{}'",
                tool_name, self.name
            )
        })?
        .with_context(|| {
            format!(
                "Failed to call tool '{}' on MCP server '{}'",
                tool_name, self.name
            )
        })?;

        let result = response
            .get("result")
            .context("MCP server did not return a result")?;

        // Check isError flag: when true, the MCP server encountered an error
        // and the content contains the error description. We propagate this
        // as an Err so the inference loop treats it as a tool failure rather
        // than passing error text to the LLM as a successful result.
        let is_error = result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let session_ends = result
            .get("_meta")
            .and_then(|m| m.get("session_ends"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let (content_parts, images) = split_content(result);

        // Bounded before anything else looks at it. A server that returns 72,141
        // characters -- measured, one file read -- takes a seventh of a 128k window in
        // one message, and every request after it carries the same.
        //
        // Before the annotation below, not after: the annotation is this client's own
        // sentence about the server's health, and cutting the middle out of a result
        // must not be able to cut that.
        let capped = tool_output::cap(&content_parts, self.max_output_chars);
        if capped.dropped > 0 {
            tracing::info!(
                "[MCP:{}] capped '{}' result: {} characters dropped, {} shown",
                self.name,
                tool_name,
                capped.dropped,
                self.max_output_chars,
            );
        }
        let content_parts = capped.text;

        // Everything this server has reported since its last result, attached to
        // this one. A tool's answer includes how well it could answer; the whole
        // argument is in `domain::tool_health`.
        //
        // On the error path too. An error is when the agent most needs to know the
        // server had already said something was wrong with it.
        let notes = match self.health.lock() {
            Ok(mut log) => log.drain(),
            Err(_) => Vec::new(),
        };
        let annotation = tool_health::annotation(&self.name, &notes);

        if is_error {
            anyhow::bail!(
                "Tool '{}' on MCP server '{}' reported an error: {}",
                tool_name,
                self.name,
                tool_health::with_annotation(content_parts, annotation),
            );
        }

        Ok((
            tool_health::with_annotation(content_parts, annotation),
            images,
            session_ends,
        ))
    }

    async fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        tracing::debug!(
            "[MCP:{}] Sending: {}",
            self.name,
            serde_json::to_string(&request).unwrap_or_default(),
        );
        self.transport.send(&request).await?;

        let response = self.read_response(id).await?;
        tracing::debug!(
            "[MCP:{}] Received: {}",
            self.name,
            serde_json::to_string(&response).unwrap_or_default()
        );

        if let Some(error) = response.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("MCP JSON-RPC error: {}", msg);
        }

        Ok(response)
    }

    /// The response to request `expected_id`, discarding anything else on the way.
    ///
    /// ## Why this checks the id
    ///
    /// It used to return the first thing it could parse, and the id it had just
    /// written was never read back. That is correct exactly as long as no request
    /// is ever abandoned -- and `tools/call` abandons one every time it times out.
    /// The dropped future stops reading; the server, which knows nothing about the
    /// timeout, finishes its work and writes the response anyway. It sits in the
    /// pipe. The next call reads it.
    ///
    /// From then on every answer belongs to the previous question, for the life of
    /// the run, and nothing detects it: the shape is valid, the content is
    /// plausible, and the id that would have given it away was not being looked
    /// at. Measured in a real run (2026-08-21, run 32436740948):
    ///
    /// ```text
    /// 01:35:11  search_issues called
    /// 01:36:11  ERROR Timed out calling tool 'search_issues'  <- client gives up
    /// 01:36:34  query "..." -> #461, #464, #408               <- server answers anyway
    /// 01:37:06  query "..." -> #408, #403, #464               <- the RETRY's answer,
    ///                                                            read by the call after it
    /// ```
    ///
    /// The agent got issues for a question it had already replaced. A wrong answer
    /// that looks like an answer is worse than the timeout that caused it.
    ///
    /// Discarding rather than resynchronising: a late response is the answer to a
    /// question nobody is waiting for any more. There is no caller to give it to.
    /// It is logged at `warn` because it means a timeout fired, which is worth
    /// seeing next to the tool that caused it.
    async fn read_response(&mut self, expected_id: u64) -> Result<Value> {
        loop {
            let value = self.transport.next_message().await?;
            match classify(&value, expected_id) {
                Incoming::TheAnswer => return Ok(value),
                Incoming::Abandoned(id) => tracing::warn!(
                    "[MCP:{}] discarding a late response to request {} while waiting for {} -- an earlier call timed out and the server answered it afterwards",
                    self.name,
                    id,
                    expected_id,
                ),
                Incoming::NotAResponse => {
                    // Bound outside the macro: `tracing`s expansion has its own
                    // `Value` trait in scope, so `Value::as_str` inside the call
                    // resolves to that one and does not compile.
                    let method = value
                        .get("method")
                        .and_then(Value::as_str)
                        .unwrap_or("?");
                    // One notification is not traffic to skip. `notifications/message`
                    // is the server reporting on itself, with a severity it chose;
                    // it used to be discarded here, which is how a degraded server
                    // stayed quiet -- see `domain::tool_health`.
                    if method == LOG_NOTIFICATION {
                        if let Some((severity, message)) = log_note(value.get("params")) {
                            let kept = match self.health.lock() {
                                Ok(mut log) => log.record(severity, &message),
                                Err(_) => false,
                            };
                            let disposition = if kept {
                                "goes out with the next result"
                            } else {
                                "already reported"
                            };
                            tracing::info!(
                                "[MCP:{}:log] {} ({})",
                                self.name,
                                message,
                                disposition,
                            );
                        }
                    } else {
                        tracing::debug!(
                            "[MCP:{}] not a response to {}, reading past it: {}",
                            self.name,
                            expected_id,
                            method,
                        );
                    }
                }
            }
        }
    }

    async fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.transport.send(&notification).await
    }

    /// `initialize`, waiting for a server that is still starting when told to.
    ///
    /// Only a refused connection is retried, and only when the caller says atoma
    /// started this server. Anything the server itself answers -- an error
    /// included -- is an answer, and is returned.
    async fn initialize(&mut self, params: Value, wait_for_it: bool) -> Result<Value> {
        loop {
            match self.send_request("initialize", params.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) if wait_for_it && not_listening_yet(&e) => {
                    tracing::debug!(
                        "[MCP:{}] not accepting connections yet, retrying: {}",
                        self.name,
                        e,
                    );
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// What a value read from a server is, relative to the request being awaited.
#[derive(Debug, PartialEq)]
enum Incoming {
    /// The response to the request in flight.
    TheAnswer,
    /// A response to a request that timed out, carrying the id it answers.
    Abandoned(u64),
    /// Nothing this client asked for: a notification, or an id it could not have
    /// issued. Read past either way.
    NotAResponse,
}

/// Whether a value read from the server answers request `expected_id`.
///
/// A function rather than a `match` inside the read loop because this is the
/// judgement that was missing, and a missing judgement is best kept somewhere a
/// test can reach. See `read_response`.
fn classify(value: &Value, expected_id: u64) -> Incoming {
    match value.get("id") {
        Some(Value::Number(n)) => match n.as_u64() {
            Some(id) if id == expected_id => Incoming::TheAnswer,
            // Includes an id this client never issued -- a negative or fractional
            // one, or a number beyond u64. It is not the answer being waited for,
            // which is the only thing that matters here.
            Some(id) => Incoming::Abandoned(id),
            // A negative, fractional, or oversized id. A server sending one is
            // broken, but it is still not the answer being waited for, and that is
            // the only question here.
            None => Incoming::NotAResponse,
        },
        // `id: null` is what a server sends when it could not parse the request
        // well enough to echo an id back. Only one request is ever in flight from
        // this client, so it is about that one: hand it to the caller, which turns
        // the `error` field into a message naming the tool.
        Some(Value::Null) => Incoming::TheAnswer,
        // A JSON-RPC notification -- `notifications/progress` and the like. Servers
        // may send these unprompted, so this is normal traffic, not a fault.
        _ => Incoming::NotAResponse,
    }
}

/// What this client tells a server it can do.
///
/// `logging` says it will listen to `notifications/message`. A server may withhold
/// those from a client that did not ask, so leaving this out is how a report of
/// degradation never arrives -- see `domain::tool_health`.
///
/// A function so a test can hold it. Nothing observable breaks when a capability
/// stops being declared: servers go quiet, tools keep answering, and the loss shows
/// up as an absence months later.
fn client_capabilities() -> Value {
    serde_json::json!({ "logging": {} })
}

/// The method name of MCP's log notification.
const LOG_NOTIFICATION: &str = "notifications/message";

/// A `notifications/message` turned into what an agent would need to read.
///
/// MCP's shape is `{ level, logger?, data }`, where `data` is any JSON. The level
/// is the server's own judgement, which is the whole reason this channel is
/// preferred over reading stderr: there is nothing to infer.
///
/// `None` for anything routine, so no caller decides that a second time. A
/// notification with no level is malformed and lands there too -- a missing
/// severity is not an urgent one.
fn log_note(params: Option<&Value>) -> Option<(Severity, String)> {
    let params = params?;
    let level = params.get("level").and_then(Value::as_str).unwrap_or("");
    let severity = tool_health::severity_of_level(level);
    if severity == Severity::Routine {
        return None;
    }

    // `data` is free-form by specification. A string is the message; anything else
    // is serialised, because a server that reports in an object is still reporting,
    // and dropping it would lose the one thing this exists to carry.
    let data = match params.get("data") {
        Some(Value::String(text)) => text.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => String::new(),
    };
    let message = match params.get("logger").and_then(Value::as_str) {
        Some(logger) if !logger.trim().is_empty() => format!("{}: {}", logger.trim(), data),
        _ => data,
    };
    Some((severity, message))
}

/// Whatever a server managed to say before it failed to start, for the error.
///
/// Generic over the handle because there are two now: over HTTP a server's stdout
/// is a log rather than the transport, and a server that dies while binding its
/// port is as likely to explain itself there as on stderr.
async fn capture_output<R>(handle: Option<R>) -> String
where
    R: AsyncRead + Unpin,
{
    let Some(mut handle) = handle else {
        return String::new();
    };
    let mut buf = String::new();
    match tokio::time::timeout(
        std::time::Duration::from_millis(500),
        handle.read_to_string(&mut buf),
    )
    .await
    {
        Ok(Ok(_)) if !buf.trim().is_empty() => buf.trim_end().to_string(),
        _ => String::new(),
    }
}

/// Both log channels, labelled, for an error message.
async fn what_it_said<E, O>(stderr: Option<E>, stdout: Option<O>) -> String
where
    E: AsyncRead + Unpin,
    O: AsyncRead + Unpin,
{
    let mut said = String::new();
    for (channel, text) in [
        ("stderr", capture_output(stderr).await),
        ("stdout", capture_output(stdout).await),
    ] {
        if !text.is_empty() {
            said.push_str(&format!("\n--- {channel} ---\n{text}\n---------------"));
        }
    }
    said
}

/// Read a channel that is not the transport: log every line, keep the ones that
/// report trouble.
///
/// Detached, and lives as long as the pipe. The connection's `Drop` kills the
/// process, which closes it.
///
/// Called once per channel that is not carrying the protocol -- stderr always, and
/// stdout as well for a server spoken to over HTTP. Both are ordinary logging there,
/// and the reason to read stdout at all is that an unread pipe stalls the writer at
/// 64KB.
fn watch_log<R>(server: String, channel: &'static str, handle: R, health: Arc<Mutex<HealthLog>>)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut reader = BufReader::new(handle);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let text = line.trim_end();
                    tracing::info!("[MCP:{}:{}] {}", server, channel, text);
                    // The fallback channel of `domain::tool_health`, and today the
                    // only one in use: no server this project ships implements
                    // `logging` yet. Severity has to be read out of the words, which
                    // is what makes this the fallback rather than the primary.
                    let severity = tool_health::severity_of_stderr(text);
                    if severity != Severity::Routine {
                        if let Ok(mut log) = health.lock() {
                            log.record(severity, text);
                        }
                    }
                }
            }
        }
    });
}

/// Whether an error means "nothing is accepting connections there yet".
///
/// Read out of the error chain rather than matched against a message: `reqwest`
/// already classifies this, and a string match would be one dependency update away
/// from silently never retrying -- which would look like a server that is slow to
/// start being a server that is broken.
fn not_listening_yet(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .is_some_and(reqwest::Error::is_connect)
    })
}

// ── Transport ─────────────────────────────────────────────────────────────────

/// How atoma reaches a server, once there is one to reach.
///
/// Two, and the difference is narrower than it looks: both carry JSON-RPC messages
/// and everything above this -- ids, timeouts, the health log, `classify` -- is
/// written once and works for either.
///
/// What genuinely differs is where a message comes from. Over stdio it is a line on
/// a pipe that may arrive at any time. Over HTTP it is in the body of the response
/// to a request, so messages arrive in batches and there is nothing to read between
/// them.
enum Transport {
    /// Pipes to a child process. `stdout` IS the transport, which is why a stdio
    /// server logs to stderr and why nothing else may read it.
    Stdio {
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    },
    /// Streamable HTTP, the transport MCP defines for a server reached over a
    /// network.
    Http(Http),
}

impl Transport {
    async fn send(&mut self, message: &Value) -> Result<()> {
        match self {
            Transport::Stdio { stdin, .. } => {
                let line = serde_json::to_string(message)?;
                stdin.write_all(format!("{line}\n").as_bytes()).await?;
                stdin.flush().await?;
                Ok(())
            }
            Transport::Http(http) => http.post(message).await,
        }
    }

    async fn next_message(&mut self) -> Result<Value> {
        match self {
            Transport::Stdio { stdout, .. } => read_json_value(stdout).await,
            Transport::Http(http) => http.next_message(),
        }
    }

    /// Remember the protocol version the server answered `initialize` with.
    ///
    /// Nothing for stdio, which has no headers to carry it.
    fn negotiated(&mut self, version: &str) {
        if let Transport::Http(http) = self {
            http.protocol = Some(version.to_string());
        }
    }
}

/// One complete JSON value from a server's stdout.
///
/// Accumulates lines because a server may pretty-print, which puts one value across
/// many lines; `is_eof` is how serde says "valid so far, incomplete".
async fn read_json_value(stdout: &mut BufReader<ChildStdout>) -> Result<Value> {
    let mut buf = String::new();
    loop {
        let mut line = String::new();
        let n = stdout
            .read_line(&mut line)
            .await
            .context("Failed to read response from MCP server")?;
        if n == 0 {
            anyhow::bail!("MCP server closed connection");
        }
        buf.push_str(&line);
        match serde_json::from_str(&buf) {
            Ok(value) => return Ok(value),
            Err(e) if e.is_eof() => {
                if buf.len() > 10_485_760 {
                    anyhow::bail!("MCP response exceeded maximum size (10MB)");
                }
                continue;
            }
            Err(e) => {
                anyhow::bail!("Failed to parse MCP JSON-RPC response: {}", e);
            }
        }
    }
}

/// A server reached over Streamable HTTP.
///
/// One POST carries one message and its answer comes back in the response body, as
/// either a single JSON object or an SSE stream that may carry notifications before
/// the answer. Both are read to completion and queued, so `next_message` hands them
/// out in the order the server sent them and `read_response` sees exactly what it
/// sees over stdio.
///
/// Read to completion rather than streamed: the specification says a server SHOULD
/// close the stream once it has sent the response, and the request timeout is what
/// bounds one that does not. Streaming would buy a notification arriving earlier
/// than the result it belongs to, and nothing here has anything to do with it any
/// sooner -- the annotation goes out with that result either way.
struct Http {
    client: reqwest::Client,
    url: String,
    /// Sent with every request. How a remote server is authenticated, and the only
    /// way -- `env` reaches a process, and there is not necessarily one.
    headers: HashMap<String, String>,
    /// The session the server assigned at `initialize`, if it assigned one. A
    /// server that works in sessions rejects a request without it.
    session: Option<String>,
    /// The version the server answered with, echoed on every later request.
    protocol: Option<String>,
    /// What the last response carried and no caller has taken yet.
    pending: VecDeque<Value>,
}

impl Http {
    fn new(url: String, headers: HashMap<String, String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            url,
            headers,
            session: None,
            protocol: None,
            pending: VecDeque::new(),
        }
    }

    async fn post(&mut self, message: &Value) -> Result<()> {
        let mut request = self
            .client
            .post(&self.url)
            .header("content-type", "application/json")
            // Both, because the server chooses which to send: one JSON object, or a
            // stream that may carry notifications ahead of the answer. A client
            // accepting only one of them refuses half the servers that exist.
            .header("accept", "application/json, text/event-stream");
        for (name, value) in &self.headers {
            request = request.header(name.as_str(), value.as_str());
        }
        if let Some(session) = &self.session {
            request = request.header("mcp-session-id", session);
        }
        if let Some(protocol) = &self.protocol {
            request = request.header("mcp-protocol-version", protocol);
        }

        let response = request
            .json(message)
            .send()
            .await
            .with_context(|| format!("Failed to reach MCP server at {}", self.url))?;

        if let Some(session) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
        {
            self.session = Some(session.to_string());
        }

        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = response.text().await.unwrap_or_default();

        // A session the server has forgotten. Said plainly because the alternative
        // reading -- a wrong url -- is a different fix, and both arrive as a 404.
        if status == reqwest::StatusCode::NOT_FOUND && self.session.is_some() {
            self.session = None;
            anyhow::bail!(
                "MCP server at {} no longer knows this session; it ended the conversation",
                self.url,
            );
        }
        if !status.is_success() {
            anyhow::bail!(
                "MCP server at {} answered {}: {}",
                self.url,
                status,
                body.trim(),
            );
        }

        // 202 with an empty body is the right answer to a notification, and there is
        // nothing to queue. Only a request expects a message back, and
        // `next_message` is where the absence of one is reported.
        for value in parse_body(&content_type, &body)? {
            self.pending.push_back(value);
        }
        Ok(())
    }

    fn next_message(&mut self) -> Result<Value> {
        self.pending
            .pop_front()
            .with_context(|| format!("MCP server at {} sent no response to the request", self.url))
    }
}

/// The JSON-RPC messages a response body carries, in the order the server sent them.
fn parse_body(content_type: &str, body: &str) -> Result<Vec<Value>> {
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    if content_type.starts_with("text/event-stream") {
        return Ok(sse_messages(body));
    }
    let value: Value = serde_json::from_str(body)
        .with_context(|| format!("Failed to parse MCP JSON-RPC response: {}", body.trim()))?;
    Ok(match value {
        // A batch. JSON-RPC allows one, and flattening it here is what lets
        // everything above this file treat one message at a time.
        Value::Array(values) => values,
        one => vec![one],
    })
}

/// The messages in a `text/event-stream` body.
///
/// One event per blank line, its `data:` lines joined with newlines as the SSE
/// specification says. `event:`, `id:` and `retry:` are read past: atoma does not
/// resume a stream, so an event id is nothing it could use.
///
/// An event that is not JSON is logged and dropped rather than failing the call. A
/// server is allowed to send things this client does not know about, and refusing
/// the whole response over one of them would turn an unknown extension into a
/// broken tool.
fn sse_messages(body: &str) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut data = String::new();
    for line in body.lines() {
        if line.is_empty() {
            end_of_event(&mut messages, &mut data);
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    // A body that ends without a trailing blank line still ends its last event.
    end_of_event(&mut messages, &mut data);
    messages
}

fn end_of_event(messages: &mut Vec<Value>, data: &mut String) {
    if data.trim().is_empty() {
        data.clear();
        return;
    }
    match serde_json::from_str(data) {
        Ok(value) => messages.push(value),
        Err(e) => tracing::warn!("[MCP] discarding an SSE event that is not JSON: {}", e),
    }
    data.clear();
}

/// Something wrong with a set of servers, found by asking them.
///
/// `fatal` is whether a run may proceed past it, and it is the only thing the two
/// callers disagree about. A guard that guards nothing does not stop a run and does
/// stop a pull request.
///
/// `message` is English for a person and `kind` is the same fact for a program. Both,
/// because neither does the other's job: an environment that wants to repair a tools
/// file cannot parse a sentence, and a person reading a log is not helped by
/// `kind=dead_guard server=files pattern=read`.
#[derive(Debug, Clone)]
pub struct Finding {
    pub fatal: bool,
    pub message: String,
    /// What is wrong and what it is wrong about, with nothing to recover from prose.
    pub kind: FindingKind,
}

/// The defects `findings` knows how to report, each carrying what a caller needs to
/// act on it.
///
/// An enum rather than a `kind` string beside a handful of optional fields, because
/// the fields are not optional: a dead guard always has a pattern, a duplicate always
/// has the servers that claim it, and a shape that cannot say otherwise is one no
/// caller has to check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FindingKind {
    /// A pattern in a server's allowlist or denylist that matches none of the tools
    /// that server advertises.
    DeadGuard {
        server: String,
        pattern: String,
        /// Every tool the server advertises -- the set the pattern failed to match.
        /// Whoever repairs the file needs to know what was there to match against;
        /// the usual cause is a `server__` prefix that the names no longer carry.
        tools: Vec<String>,
    },
    /// One tool name claimed by more than one server, which routing cannot resolve.
    DuplicateTool {
        tool: String,
        /// The servers claiming it, in the order the tools file declares them.
        servers: Vec<String>,
    },
}

/// What every machine-readable configuration line starts with.
///
/// A constant because it is the token a caller greps for: it may not drift by a typo
/// in one `format!` while the documentation says something else.
pub const CONFIG_FINDING_LINE: &str = "ATOMA_CONFIG_FINDING:";

impl Finding {
    /// This finding as one line of fields, addressed to the environment running atoma
    /// rather than to a person.
    ///
    /// ```text
    /// ATOMA_CONFIG_FINDING: kind=dead_guard severity=warn server=files_ro
    ///                       pattern=read tools=read,grep,glob
    /// ATOMA_CONFIG_FINDING: kind=duplicate_tool severity=error tool=read
    ///                       servers=files,files_ro
    /// ```
    ///
    /// (wrapped here to fit; each is one line.)
    ///
    /// Same shape as `ATOMA_TOKEN_USAGE` and `ATOMA_INFERENCE_USAGE`: an `ATOMA_`
    /// name, a colon, then space-separated `key=value`.
    ///
    /// **The fields are the contract and the sentence in `message` is not.** A caller
    /// that greps the English is a caller whose tooling atoma breaks by rewording a
    /// warning -- so the prose stays free to be good English for a person, and
    /// anything acting on a finding reads `kind=` and the fields beside it.
    ///
    /// `severity` is atoma's own verdict about the configuration -- a dead guard is
    /// `warn`, a name two servers claim is `error` -- and not the caller's policy
    /// about it. `--fail-on-tool-findings` changes what a run does with a
    /// `severity=warn` line; it does not change what the line says, because the
    /// caller that set the flag is the one caller that already knows it did.
    ///
    /// Values are percent-encoded for whitespace, `,` and `%`, and written literally
    /// otherwise. Server names and glob patterns come out of the caller's tools file,
    /// so a space in one is possible in principle, and one space would turn a value
    /// into two fields for every reader at once. `,` is encoded because it separates
    /// the entries of a list, and `%` because encoding anything at all makes it the
    /// escape character. Ordinary names and globs contain none of the three and
    /// survive byte for byte.
    pub fn machine_line(&self) -> String {
        let severity = if self.fatal { "error" } else { "warn" };
        match &self.kind {
            FindingKind::DeadGuard {
                server,
                pattern,
                tools,
            } => format!(
                "{} kind=dead_guard severity={} server={} pattern={} tools={}",
                CONFIG_FINDING_LINE,
                severity,
                field(server),
                field(pattern),
                field_list(tools),
            ),
            FindingKind::DuplicateTool { tool, servers } => format!(
                "{} kind=duplicate_tool severity={} tool={} servers={}",
                CONFIG_FINDING_LINE,
                severity,
                field(tool),
                field_list(servers),
            ),
        }
    }
}

/// One value, with the three characters that would break the line's own grammar
/// percent-encoded as their UTF-8 bytes.
///
/// Whitespace would split one field into two, a comma would split one list entry into
/// two, and `%` has to be encoded for either of those to be reversible. Everything
/// else is written as it is, including the `_` and `*` that real names and globs are
/// made of: encoding those would make the common case unreadable to buy nothing.
fn field(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '%' => out.push_str("%25"),
            ',' => out.push_str("%2C"),
            c if c.is_whitespace() => {
                let mut buf = [0u8; 4];
                for byte in c.encode_utf8(&mut buf).as_bytes() {
                    out.push_str(&format!("%{:02X}", byte));
                }
            }
            c => out.push(c),
        }
    }
    out
}

/// Several values as one comma-separated field. Empty is written as nothing, which is
/// a server that advertises no tools at all -- the state in which every pattern it
/// declares is dead.
fn field_list(values: &[String]) -> String {
    let mut out = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&field(value));
    }
    out
}

/// What these servers, having said what they have, are wrong about.
///
/// Pure, and the only place either check lives, so registration and
/// `atoma validate --with-live-tools` cannot come to different conclusions about the
/// same tools file. The one before it could not exist at all: until a server has
/// answered, a pattern is a string and a name is a guess.
pub fn findings(configs: &[ToolDef], offered: &[(String, Vec<RegisteredTool>)]) -> Vec<Finding> {
    let mut out = Vec::new();
    let mut claimed: HashMap<String, String> = HashMap::new();

    for (config, (_, tools)) in configs.iter().zip(offered.iter()) {
        // Before the access filter, not after: a pattern is dead because it matches
        // nothing the server has, and the filter removing a tool is that pattern
        // working.
        let names: Vec<String> = tools.iter().map(|t| t.prefixed_name.clone()).collect();
        for pattern in hooks::unmatched_patterns(&config.hooks, &names) {
            out.push(Finding {
                fatal: false,
                message: format!(
                    "[{}] allow/deny pattern '{}' matches none of this server's tools \
                     ({}). It is guarding nothing.",
                    config.name,
                    pattern,
                    names.join(", "),
                ),
                kind: FindingKind::DeadGuard {
                    server: config.name.clone(),
                    pattern: pattern.clone(),
                    tools: names.clone(),
                },
            });
        }

        // After it, here: a tool a server denies is not a name it claims, and counting
        // it would report a clash that cannot happen.
        for tool in tools {
            if hooks::access_denial_reason(&config.hooks, &tool.prefixed_name).is_some() {
                continue;
            }
            if let Some(first) = claimed.get(&tool.prefixed_name) {
                out.push(Finding {
                    fatal: true,
                    message: format!(
                        "Two servers offer a tool named '{}': '{}' and '{}'. A server \
                         with `unprefixed: true` gives its tools their own names, so two \
                         of them cannot offer the same one. Take the flag off one, or \
                         deny the tool on one.",
                        tool.prefixed_name, first, config.name,
                    ),
                    kind: FindingKind::DuplicateTool {
                        tool: tool.prefixed_name.clone(),
                        servers: vec![first.clone(), config.name.clone()],
                    },
                });
                continue;
            }
            claimed.insert(tool.prefixed_name.clone(), config.name.clone());
        }
    }

    // Fatal first, so a caller that stops at the first one stops at the worst.
    out.sort_by_key(|finding| !finding.fatal);
    out
}

/// Start every server, ask what it has, and say what is wrong. Nothing is kept.
///
/// For a caller that wants the answer without running anything -- `atoma validate
/// --with-live-tools`. A server that will not start, or will not answer inside its
/// own timeout, is an error rather than a server with no tools: a check whose input
/// never arrived has not passed.
pub async fn inspect(configs: &[ToolDef]) -> Result<Vec<Finding>> {
    let mut offered = Vec::new();
    for config in configs {
        let mut conn = McpConnection::connect(config)
            .await
            .with_context(|| format!("MCP server '{}' did not start", config.name))?;
        let tools = conn
            .list_tools()
            .await
            .with_context(|| format!("MCP server '{}' did not answer tools/list", config.name))?;
        offered.push((config.name.clone(), tools));
    }
    Ok(findings(configs, &offered))
}

/// What a set of findings makes a run say, and then do.
///
/// The fields are in the order they happen, and that order is the whole point: every
/// machine-readable line is written before anything reads `refusal`. A fatal finding
/// ends the run before any JSON result envelope exists, so its
/// `ATOMA_CONFIG_FINDING` line is the ONLY channel the caller has -- emitting it after
/// the refusal would mean the worst finding is the one nothing outside the log ever
/// hears about.
struct FindingReport {
    /// One `ATOMA_CONFIG_FINDING` line per finding, fatal ones included, each paired
    /// with whether that finding is fatal.
    ///
    /// Paired rather than left parallel to the findings themselves, because the pairing
    /// is what decides the log level, and a level taken from the wrong element is a
    /// line that disappears at exactly the severity it was written for.
    lines: Vec<(bool, String)>,
    /// The prose a person reads, which stops at the first fatal finding exactly as it
    /// did when one loop did both jobs: `findings` sorts fatal first, so a run that is
    /// about to be refused does not also list what it would have tolerated.
    warnings: Vec<String>,
    /// What to fail with, or `None` to go on.
    refusal: Option<String>,
}

impl FindingReport {
    /// Write all of it, in the order that matters, and hand back what to fail with.
    ///
    /// The level follows the finding. Every line was `info` at first, which inverted
    /// the intent: under `RUST_LOG=warn` -- ordinary in CI -- the prose still reached
    /// stderr through the refusal, while the machine-readable line, the one channel a
    /// fatal finding has before any result envelope exists, was filtered away.
    fn emit(self) -> Option<String> {
        for (fatal, line) in &self.lines {
            if *fatal {
                tracing::error!("{}", line);
            } else {
                tracing::warn!("{}", line);
            }
        }
        for message in &self.warnings {
            tracing::warn!("{}", message);
        }
        self.refusal
    }
}

/// Decide the above, apart from doing any of it.
///
/// Separated from `from_configs` because the order is the contract and `from_configs`
/// needs a started server for every entry in a tools file, so nothing there is
/// reachable from a unit test. Here the whole decision is a function of the findings.
fn report_on(found: &[Finding], fail_on_findings: bool) -> FindingReport {
    let mut report = FindingReport {
        lines: found
            .iter()
            .map(|finding| (finding.fatal, finding.machine_line()))
            .collect(),
        warnings: Vec::new(),
        refusal: None,
    };

    for finding in found {
        if finding.fatal {
            report.refusal = Some(finding.message.clone());
            return report;
        }
        report.warnings.push(finding.message.clone());
    }

    // Nothing above this line is a run-stopper, so this is the only thing the flag
    // decides. See `Command::Run::fail_on_tool_findings` for why the default is off.
    if fail_on_findings && !found.is_empty() {
        report.refusal = Some(format!(
            "{} tool configuration finding(s), and --fail-on-tool-findings asks for a \
             run to stop on them: {}",
            found.len(),
            report.warnings.join(" "),
        ));
    }

    report
}

/// Manages multiple MCP connections and routes tool calls by tool prefix.
pub struct McpRegistry {
    connections: HashMap<String, McpConnection>,
    /// Tool name to where it goes. Replaces splitting the name on `__`.
    routes: HashMap<String, Route>,
    tools: Vec<RegisteredTool>,
    hooks: HashMap<String, Hooks>,
}

impl McpRegistry {
    /// Start every declared server and build the registry.
    ///
    /// `fail_on_findings` is the caller's answer to a configuration defect that does
    /// not by itself stop a run -- `--fail-on-tool-findings`. `false` is what every
    /// caller got before the flag existed: warn, emit the machine-readable line, and
    /// run. Whatever it is, every finding's line is written before anything is
    /// refused.
    pub async fn from_configs(configs: &[ToolDef], fail_on_findings: bool) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        for config in configs {
            if !seen.insert(&config.name) {
                anyhow::bail!(
                    "Duplicate MCP server name: '{}'. Each server must have a unique name.",
                    config.name,
                );
            }
        }

        // Gather first, then check, then build. The checks need every server's answer,
        // and `findings` is what `atoma validate --with-live-tools` runs over the same
        // data -- so neither can be right while the other is wrong.
        let mut connections = HashMap::new();
        let mut offered: Vec<(String, Vec<RegisteredTool>)> = Vec::new();
        for config in configs {
            let mut conn = McpConnection::connect(config).await?;
            let tools = conn.list_tools().await?;
            connections.insert(config.name.clone(), conn);
            offered.push((config.name.clone(), tools));
        }

        // The lines first, then the prose, then the decision. `FindingReport::emit`
        // owns that order; `report_on` says what `fail_on_findings` changes about it.
        let report = report_on(&findings(configs, &offered), fail_on_findings);
        if let Some(message) = report.emit() {
            anyhow::bail!("{}", message);
        }

        let mut all_tools = Vec::new();
        let mut routes: HashMap<String, Route> = HashMap::new();
        for (config, (_, tools)) in configs.iter().zip(offered) {
            for tool in tools {
                if hooks::access_denial_reason(&config.hooks, &tool.prefixed_name).is_some() {
                    continue;
                }
                routes.insert(
                    tool.prefixed_name.clone(),
                    Route {
                        server: config.name.clone(),
                        tool: tool.tool_name.clone(),
                    },
                );
                all_tools.push(tool);
            }
        }
        let hooks: HashMap<String, Hooks> = configs
            .iter()
            .map(|c| (c.name.clone(), c.hooks.clone()))
            .collect();

        // Both lists together is allowed -- the denylist is checked first -- and worth
        // saying out loud once, here, rather than refused as it used to be.
        for (name, h) in &hooks {
            hooks::describe_hooks(name, h);
        }

        Ok(McpRegistry {
            connections,
            routes,
            tools: all_tools,
            hooks,
        })
    }

    /// Return OpenAI-compatible tool definitions for all registered tools.
    pub fn tool_definitions(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                let mut def = serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.prefixed_name,
                        "description": t.schema.get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or(""),
                    }
                });
                if let Some(input_schema) = t.schema.get("inputSchema") {
                    def["function"]["parameters"] = input_schema.clone();
                }
                def
            })
            .collect()
    }

    /// Call a tool by its prefixed name, running access-control hooks.
    ///
    /// Order: denylist/allowlist → before_tool hooks → MCP call → after_tool hooks.
    ///
    /// Each list holds the tools file's file-wide hooks first and the server's own
    /// after them, concatenated at load time. A before-hook that refuses ends the call
    /// and the rest do not run. Every after-hook runs, and whatever they have to say is
    /// appended to the result -- see `domain::tool::Hooks::after_tool` for why that is
    /// where it goes.
    pub async fn call_tool_with_hooks(
        &mut self,
        agent_name: &str,
        prefixed_name: &str,
        arguments: &Value,
    ) -> Result<crate::domain::ports::ToolCallResult> {
        // The route, not the name. `self.hooks` is keyed by server name, and splitting
        // the call's name on `__` only recovers one for a PREFIXED server: with
        // `unprefixed: true` the prefixed name IS the bare tool name, so
        // `"read_text_file".split("__").next()` answered `"read_text_file"`, no server
        // was ever found under it, and every hook that server declared -- including its
        // denylist and allowlist -- was skipped without a word. `routes` already holds
        // the answer, put there by the same loop that registered the tool.
        let hooks = self
            .routes
            .get(prefixed_name)
            .and_then(|route| self.hooks.get(&route.server))
            .cloned();

        if let Some(ref h) = hooks {
            hooks::check_access(h, prefixed_name)?;

            for script in &h.before_tool {
                let payload = serde_json::json!({
                    "agent": agent_name,
                    "tool": prefixed_name,
                    "arguments": arguments,
                });
                hooks::run_before_hook(script, payload).await?;
            }
        }

        let (mut content, images, session_ends) = self.call_tool(prefixed_name, arguments).await?;

        if let Some(ref h) = hooks {
            for script in &h.after_tool {
                let payload = serde_json::json!({
                    "agent": agent_name,
                    "tool": prefixed_name,
                    "arguments": arguments,
                    "result": content,
                });
                if let Some(notice) = hooks::run_after_hook(script, payload).await {
                    // Appended rather than replacing anything: the result is the answer
                    // the agent asked for, and the notice is a second thing to know. A
                    // server that reports a problem about itself already arrives this way,
                    // so the shape is one the agent has been told how to read.
                    content.push_str(&format!("\n\n{}", notice));
                }
            }
        }

        Ok(crate::domain::ports::ToolCallResult {
            content,
            images,
            session_ends,
        })
    }

    pub(crate) async fn call_tool(
        &mut self,
        prefixed_name: &str,
        arguments: &Value,
    ) -> Result<(String, Vec<Value>, bool)> {
        // The table, not the name: a tool on a server that set `unprefixed` has no
        // `__` to split, and splitting one that does would route `read_text_file` on a
        // server called `filesystem` to a server called `filesystem_readonly` the day
        // somebody names one with an underscore pair in it.
        let route = match self.routes.get(prefixed_name) {
            Some(route) => route.clone(),
            // Before the unknown-tool complaint, because "unknown tool" is true and
            // useless to a caller that meant a skill. See `skill_called_as_tool_message`.
            None => {
                if let Some(message) =
                    crate::domain::skill::skill_called_as_tool_message(prefixed_name)
                {
                    anyhow::bail!(message);
                }
                anyhow::bail!(
                    "Unknown tool '{prefixed_name}'. A tool is named as its server \
                     advertises it, with `server__` in front unless that server sets \
                     `unprefixed: true`."
                );
            }
        };
        let tool_name = route.tool.as_str();

        let conn = self
            .connections
            .get_mut(&route.server)
            .with_context(|| format!("Unknown MCP server: {}", route.server))?;

        conn.call_tool(tool_name, arguments).await
    }
}

// ── Port implementation ───────────────────────────────────────────────────────

#[async_trait::async_trait]
impl crate::domain::ports::ToolPort for McpRegistry {
    fn tool_definitions(&self) -> Vec<serde_json::Value> {
        self.tool_definitions()
    }

    async fn call_tool(
        &mut self,
        agent_name: &str,
        prefixed_name: &str,
        arguments: &serde_json::Value,
    ) -> anyhow::Result<crate::domain::ports::ToolCallResult> {
        self.call_tool_with_hooks(agent_name, prefixed_name, arguments)
            .await
    }
}

// ── MCP factory ───────────────────────────────────────────────────────────────

/// Factory adapter implementing `McpFactory` for constructing `McpRegistry`.
pub struct McpRegistryFactory {
    /// Whether a finding that only warns should stop the run -- the
    /// `--fail-on-tool-findings` flag, carried here rather than through `McpFactory`.
    ///
    /// The port's `build` takes the tool definitions and nothing else, and that
    /// signature is implemented by every test double in the suite. This is a policy
    /// the adapter holds, not a second thing every caller of the port has to answer.
    fail_on_findings: bool,
}

impl McpRegistryFactory {
    /// Off is the default and is what `Default` would have given, but there is no
    /// `Default` here: the one caller that builds this is the one that read the flag,
    /// and a factory built without saying which policy it carries is a factory whose
    /// policy nobody stated.
    pub fn new(fail_on_findings: bool) -> Self {
        Self { fail_on_findings }
    }
}

#[async_trait::async_trait]
impl crate::domain::ports::McpFactory for McpRegistryFactory {
    async fn build(
        &self,
        tool_defs: &[crate::domain::tool::ToolDef],
    ) -> anyhow::Result<Box<dyn crate::domain::ports::ToolPort + Send>> {
        let registry = McpRegistry::from_configs(tool_defs, self.fail_on_findings).await?;
        Ok(Box::new(registry))
    }
}

#[cfg(test)]
mod split_content_tests {
    use super::split_content;
    use serde_json::json;

    // A picture used to reach the model as `serde_json::to_string` of the whole
    // block: thousands of base64 characters it could only read as characters.
    #[test]
    fn an_image_leaves_the_text_and_travels_on_its_own() {
        let result = json!({"content": [
            {"type": "text", "text": "Here is the screen:"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
        ]});
        let (text, images) = split_content(&result);
        assert_eq!(text, "Here is the screen:\n[image]");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0]["data"], "AAAA");
        assert!(
            !text.contains("AAAA"),
            "base64 must not be left in the text"
        );
    }

    #[test]
    fn a_text_only_result_is_unchanged_and_carries_no_images() {
        let result = json!({"content": [{"type": "text", "text": "done"}]});
        assert_eq!(split_content(&result), ("done".to_string(), Vec::new()));
    }

    // An unknown block type is not a reason to lose it.
    #[test]
    fn an_unknown_block_is_still_serialised_into_the_text() {
        let result = json!({"content": [{"type": "audio", "data": "BBBB"}]});
        let (text, images) = split_content(&result);
        assert!(text.contains("audio"));
        assert!(images.is_empty());
    }

    #[test]
    fn a_result_without_content_falls_back_to_the_whole_value() {
        let result = json!({"unexpected": true});
        let (text, images) = split_content(&result);
        assert!(text.contains("unexpected"));
        assert!(images.is_empty());
    }
}

#[cfg(test)]
mod classify_tests {
    use super::{classify, Incoming};
    use serde_json::json;

    /// The case that was broken. A `tools/call` timed out, the server finished and
    /// wrote its answer anyway, and the next call read it -- so every answer from
    /// then on belonged to the previous question. Nothing detected it, because the
    /// id was written and never read back.
    #[test]
    fn a_late_response_to_an_abandoned_request_is_not_the_answer() {
        let stale = json!({"jsonrpc": "2.0", "id": 7, "result": {"content": []}});
        assert_eq!(classify(&stale, 8), Incoming::Abandoned(7));
    }

    #[test]
    fn the_response_with_the_matching_id_is_the_answer() {
        let fresh = json!({"jsonrpc": "2.0", "id": 8, "result": {"content": []}});
        assert_eq!(classify(&fresh, 8), Incoming::TheAnswer);
    }

    /// A server may send these at any time, so they are read past rather than
    /// treated as an answer or as a fault.
    #[test]
    fn a_notification_has_no_id_and_answers_nothing() {
        let progress = json!({"jsonrpc": "2.0", "method": "notifications/progress"});
        assert_eq!(classify(&progress, 8), Incoming::NotAResponse);
    }

    /// The one case where a mismatched id must still be delivered: the server
    /// could not parse the request well enough to echo an id, so it sent `null`.
    /// Skipping it would wait out the whole timeout for a reply already received.
    #[test]
    fn a_null_id_carries_an_error_about_the_request_in_flight() {
        let parse_error = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700}});
        assert_eq!(classify(&parse_error, 8), Incoming::TheAnswer);
    }

    /// An id this client cannot have issued. Not the answer being waited for,
    /// which is all this needs to decide.
    #[test]
    fn an_id_that_is_not_a_u64_is_never_the_answer() {
        for id in [json!(-1), json!(1.5)] {
            let value = json!({"jsonrpc": "2.0", "id": id, "result": {}});
            assert_ne!(classify(&value, 8), Incoming::TheAnswer, "{id}");
        }
    }

    /// Reading past a stale response has to reach the real one, however many are
    /// queued: two consecutive timeouts leave two answers in the pipe.
    #[test]
    fn several_stale_responses_are_all_read_past() {
        let queued = [
            json!({"id": 5, "result": {}}),
            json!({"id": 6, "result": {}}),
        ];
        for value in &queued {
            assert!(matches!(classify(value, 7), Incoming::Abandoned(_)));
        }
        assert_eq!(
            classify(&json!({"id": 7, "result": {}}), 7),
            Incoming::TheAnswer
        );
    }
}

#[cfg(test)]
mod log_note_tests {
    use super::{log_note, LOG_NOTIFICATION};
    use crate::domain::tool_health::Severity;
    use serde_json::json;

    /// The protocol channel, and why it is the primary one: the server said how bad
    /// it was, so nothing has to be guessed from the words.
    #[test]
    fn a_warning_carries_the_servers_own_severity() {
        let params = json!({"level": "warning", "data": "reranker unavailable"});
        assert_eq!(
            log_note(Some(&params)),
            Some((Severity::Warning, "reranker unavailable".to_string())),
        );
    }

    /// Routine traffic is not the agent's business. Dropped here so no caller has to
    /// decide it a second time.
    #[test]
    fn an_info_message_is_not_a_report() {
        let params = json!({"level": "info", "data": "listening"});
        assert_eq!(log_note(Some(&params)), None);
    }

    /// A missing severity is not an urgent one -- and treating it as urgent would put
    /// a malformed server's chatter into every result.
    #[test]
    fn a_notification_without_a_level_is_dropped() {
        assert_eq!(log_note(Some(&json!({"data": "something"}))), None);
        assert_eq!(log_note(None), None);
    }

    #[test]
    fn the_logger_name_is_kept_when_there_is_one() {
        let params = json!({"level": "error", "logger": "search.index", "data": "no such path"});
        let (severity, message) = log_note(Some(&params)).expect("an error is a report");
        assert_eq!(severity, Severity::Error);
        assert_eq!(message, "search.index: no such path");
    }

    /// `data` is free-form by specification. A server reporting in an object is still
    /// reporting; dropping it would lose the one thing this exists to carry.
    #[test]
    fn structured_data_survives_as_text() {
        let params = json!({"level": "warning", "data": {"stage": "rerank", "fell_back": true}});
        let (_, message) = log_note(Some(&params)).unwrap();
        assert!(message.contains("rerank"), "{message}");
        assert!(message.contains("fell_back"), "{message}");
    }

    /// A report with no `data` at all still reaches the agent as its level, rather
    /// than being lost for having said nothing.
    #[test]
    fn a_bare_level_is_still_a_report() {
        let (severity, message) = log_note(Some(&json!({"level": "error"}))).unwrap();
        assert_eq!(severity, Severity::Error);
        assert_eq!(
            message, "",
            "empty here; `HealthLog::record` is what refuses it"
        );
    }

    /// Pinned because the string is the protocol's, not this project's: a typo would
    /// silently return the code to discarding every report.
    #[test]
    fn the_method_name_is_the_one_mcp_defines() {
        assert_eq!(LOG_NOTIFICATION, "notifications/message");
    }

    /// The declaration a server reads before it decides whether to report anything.
    /// Undeclaring it breaks nothing that fails: servers just go quiet.
    #[test]
    fn the_client_says_it_will_listen_to_log_notifications() {
        let capabilities = super::client_capabilities();
        assert!(
            capabilities.get("logging").is_some(),
            "a server may send nothing to a client that did not ask: {capabilities}",
        );
    }
}

#[cfg(test)]
mod http_body_tests {
    use super::{parse_body, sse_messages};
    use serde_json::json;

    const JSON: &str = "application/json";
    const SSE: &str = "text/event-stream";

    #[test]
    fn one_json_object_is_one_message() {
        let body = r#"{"jsonrpc":"2.0","id":1,"result":{}}"#;
        let messages = parse_body(JSON, body).expect("valid json");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["id"], json!(1));
    }

    /// JSON-RPC allows a batch, and flattening it here is what lets everything
    /// above this file go on handling one message at a time.
    #[test]
    fn a_batch_is_flattened_in_order() {
        let body = r#"[{"id":1,"result":{}},{"id":2,"result":{}}]"#;
        let messages = parse_body(JSON, body).expect("valid json");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["id"], json!(1));
        assert_eq!(messages[1]["id"], json!(2));
    }

    /// The right answer to a notification is 202 with nothing in it. Not an error:
    /// nothing was asked.
    #[test]
    fn an_empty_body_carries_nothing() {
        assert!(parse_body(JSON, "").expect("empty is fine").is_empty());
        assert!(parse_body(SSE, "   \n").expect("blank is fine").is_empty());
    }

    #[test]
    fn a_body_that_is_not_json_is_an_error_naming_the_body() {
        let message = parse_body(JSON, "<html>gateway timeout</html>")
            .expect_err("not json")
            .to_string();
        assert!(message.contains("gateway timeout"), "{message}");
    }

    /// The case this exists for: the notification arrives ahead of the result it
    /// belongs to, on the same stream. That ordering is what `read_response` reads
    /// past to find its answer, and what `domain::tool_health` then attaches.
    #[test]
    fn a_stream_carries_a_notification_then_the_answer() {
        let warning =
            r#"{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"warning"}}"#;
        let answer = r#"{"jsonrpc":"2.0","id":7,"result":{}}"#;
        let body = format!("event: message\r\ndata: {warning}\r\n\r\ndata: {answer}\r\n\r\n");
        let messages = parse_body(SSE, &body).expect("a stream parses");
        assert_eq!(messages.len(), 2, "{messages:?}");
        assert_eq!(messages[0]["method"], json!("notifications/message"));
        assert_eq!(messages[1]["id"], json!(7));
    }

    /// A `data:` field split across lines is one value joined with newlines, which
    /// is the SSE specification and also how a server pretty-prints.
    #[test]
    fn several_data_lines_are_one_event() {
        let body = "data: {\ndata: \"id\": 3,\ndata: \"result\": {}\ndata: }\n\n";
        let messages = sse_messages(body);
        assert_eq!(messages.len(), 1, "{messages:?}");
        assert_eq!(messages[0]["id"], json!(3));
    }

    /// A body that ends without a trailing blank line still ends its last event.
    /// Servers do this, and losing the answer to the last request of a run would be
    /// the hardest kind of bug to find.
    #[test]
    fn the_last_event_needs_no_blank_line() {
        let messages = sse_messages("data: {\"id\":9,\"result\":{}}");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["id"], json!(9));
    }

    /// Comments, ids and retry hints are read past. atoma does not resume a stream,
    /// so an event id is nothing it could use.
    #[test]
    fn the_fields_this_client_cannot_use_are_ignored() {
        let event = r#"{"id":1,"result":{}}"#;
        let body = format!(": keep-alive\nid: 42\nretry: 3000\nevent: message\ndata: {event}\n\n");
        let messages = sse_messages(&body);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["id"], json!(1));
    }

    /// One unreadable event does not fail the call. A server may send things this
    /// client knows nothing about, and refusing the whole response over one of them
    /// would turn an unknown extension into a broken tool.
    #[test]
    fn an_event_that_is_not_json_is_dropped_and_the_rest_survive() {
        let body = "data: not json at all\n\ndata: {\"id\":2,\"result\":{}}\n\n";
        let messages = sse_messages(body);
        assert_eq!(messages.len(), 1, "{messages:?}");
        assert_eq!(messages[0]["id"], json!(2));
    }
}

#[cfg(test)]
mod finding_line_tests {
    use super::{field, findings, report_on, Finding, FindingKind, RegisteredTool};
    use crate::domain::tool::{Hooks, ToolDef};
    use std::collections::HashMap;

    /// `unprefixed`, because that is the arrangement both defects live in: the tools
    /// keep their own names, so a `server__` pattern guards nothing and two servers
    /// can claim one name.
    fn server(name: &str, allow: &[&str], deny: &[&str]) -> ToolDef {
        ToolDef {
            name: name.to_string(),
            command: "/bin/true".to_string(),
            args: Vec::new(),
            env: HashMap::new(),
            url: None,
            headers: HashMap::new(),
            hooks: Hooks {
                tool_allowlist: allow.iter().map(|s| s.to_string()).collect(),
                tool_denylist: deny.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            },
            unprefixed: true,
            max_output_chars: None,
            request_timeout_secs: None,
        }
    }

    fn advertised(names: &[&str]) -> Vec<RegisteredTool> {
        names
            .iter()
            .map(|name| RegisteredTool {
                prefixed_name: name.to_string(),
                tool_name: name.to_string(),
                schema: serde_json::json!({}),
            })
            .collect()
    }

    fn dead_guard() -> Finding {
        Finding {
            fatal: false,
            message: "a pattern that is guarding nothing".to_string(),
            kind: FindingKind::DeadGuard {
                server: "files_ro".to_string(),
                pattern: "files_ro__*".to_string(),
                tools: vec!["read".to_string(), "grep".to_string()],
            },
        }
    }

    fn duplicate_tool() -> Finding {
        Finding {
            fatal: true,
            message: "two servers offer a tool named 'read'".to_string(),
            kind: FindingKind::DuplicateTool {
                tool: "read".to_string(),
                servers: vec!["files".to_string(), "files_ro".to_string()],
            },
        }
    }

    /// Pinned whole, deliberately. This line is a contract with whatever is running
    /// atoma, and a contract asserted field by field is one a refactor can reword a
    /// piece of without a test saying so.
    #[test]
    fn a_dead_guard_line_names_the_server_the_pattern_and_every_tool() {
        let configs = [server("files_ro", &[], &["files_ro__*"])];
        let offered = [("files_ro".to_string(), advertised(&["read", "grep"]))];
        let found = findings(&configs, &offered);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(
            found[0].machine_line(),
            "ATOMA_CONFIG_FINDING: kind=dead_guard severity=warn server=files_ro \
             pattern=files_ro__* tools=read,grep",
        );
    }

    #[test]
    fn a_duplicate_tool_line_names_the_tool_and_both_servers() {
        let configs = [server("files", &[], &[]), server("files_ro", &[], &[])];
        let offered = [
            ("files".to_string(), advertised(&["read"])),
            ("files_ro".to_string(), advertised(&["read"])),
        ];
        let found = findings(&configs, &offered);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(
            found[0].machine_line(),
            "ATOMA_CONFIG_FINDING: kind=duplicate_tool severity=error tool=read \
             servers=files,files_ro",
        );
    }

    /// The case the line exists for. A fatal finding ends the run before any result
    /// envelope is written, so a line emitted after the refusal would not be emitted
    /// at all -- and the worst defect would be the one only a log could tell anyone
    /// about.
    #[test]
    fn a_fatal_finding_emits_its_line_before_the_run_is_refused() {
        let report = report_on(&[duplicate_tool()], false);
        assert_eq!(report.lines.len(), 1, "{:?}", report.lines);
        // Destructured rather than indexed twice: the flag is what now picks the log
        // level, so a test that reads the line without reading the flag would pass
        // while the line went out at the wrong severity.
        let (fatal, line) = &report.lines[0];
        assert!(*fatal, "a duplicate tool name is fatal");
        assert!(
            line.contains("kind=duplicate_tool severity=error"),
            "{:?}",
            report.lines,
        );
        assert!(report.refusal.is_some(), "a duplicate name is still fatal");
    }

    /// Off is what every caller had before the flag existed, and on changes what the
    /// run does rather than what it says.
    #[test]
    fn a_warning_stops_a_run_only_when_the_caller_asks_it_to() {
        let found = [dead_guard()];
        let tolerated = report_on(&found, false);
        let refused = report_on(&found, true);
        assert!(tolerated.refusal.is_none(), "{:?}", tolerated.refusal);
        assert!(refused.refusal.is_some());
        assert_eq!(
            tolerated.warnings, refused.warnings,
            "the prose a person reads is the same either way",
        );
        assert_eq!(
            tolerated.lines, refused.lines,
            "the flag is the caller's policy, not a different finding",
        );
        assert!(
            refused.lines[0].1.contains("severity=warn"),
            "severity is atoma's verdict and not the caller's: {:?}",
            refused.lines,
        );
    }

    /// A run with nothing wrong is not refused even by a caller that asked to be
    /// strict, so the flag cannot turn a clean configuration into a failure.
    #[test]
    fn no_findings_is_no_lines_and_no_refusal() {
        let report = report_on(&[], true);
        assert!(report.lines.is_empty());
        assert!(report.warnings.is_empty());
        assert!(report.refusal.is_none());
    }

    /// Server names and glob patterns come out of the caller's file, and one space in
    /// one of them would turn a value into two fields for every reader at once.
    #[test]
    fn a_value_with_a_space_in_it_cannot_become_two_fields() {
        let finding = Finding {
            fatal: false,
            message: "a server whose name has a space in it".to_string(),
            kind: FindingKind::DeadGuard {
                server: "my files".to_string(),
                pattern: "read *".to_string(),
                tools: vec!["read".to_string()],
            },
        };
        let line = finding.machine_line();
        assert_eq!(
            line.split_whitespace().count(),
            6,
            "the name and one token per field: {line}",
        );
        assert!(line.contains("server=my%20files"), "{line}");
        assert!(line.contains("pattern=read%20*"), "{line}");
    }

    /// `%` is encoded because encoding anything at all makes it the escape character,
    /// and `,` because it is what separates the entries of a list.
    #[test]
    fn the_escape_character_and_the_list_separator_are_themselves_encoded() {
        assert_eq!(field("100%"), "100%25");
        assert_eq!(field("a,b"), "a%2Cb");
        assert_eq!(field("read_text_file"), "read_text_file");
    }

    /// A server that advertises nothing is the state in which every pattern it
    /// declares is dead, so the empty list is a value a reader will meet.
    #[test]
    fn a_server_that_advertises_no_tools_writes_an_empty_list() {
        let configs = [server("silent", &[], &["silent__*"])];
        let offered = [("silent".to_string(), advertised(&[]))];
        let found = findings(&configs, &offered);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].machine_line().ends_with(" tools="), "{found:?}");
    }
}
