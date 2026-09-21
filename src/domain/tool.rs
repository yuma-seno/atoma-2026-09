use std::collections::HashMap;

/// Hook configuration for a tool server.
///
/// Defines access control and lifecycle scripts for tool calls.
#[derive(Debug, Clone, Default)]
pub struct Hooks {
    /// Glob patterns for allowed tools. Empty = all tools allowed.
    pub tool_allowlist: Vec<String>,
    /// Glob patterns for blocked tools. Checked before the allowlist.
    pub tool_denylist: Vec<String>,
    /// Scripts invoked before each tool call, in order.
    ///
    /// Receives JSON on stdin: `{"agent": "...", "tool": "...", "arguments": {...}}`
    /// Must respond with JSON: `{"allow": true}` or `{"allow": false, "reason": "..."}`
    /// Non-zero exit or invalid JSON is treated as a deny (fail-closed).
    ///
    /// A list rather than one script because a tools file may declare hooks that apply
    /// to every server as well as hooks that apply to one. They are concatenated at
    /// load time, the file-wide ones first, so nothing downstream has to know which
    /// kind it is running. The first refusal wins and the rest do not run.
    pub before_tool: Vec<String>,
    /// Scripts invoked after each successful tool call, in order.
    ///
    /// Receives JSON on stdin: `{"agent": "...", "tool": "...", "arguments": {...}, "result": "..."}`
    ///
    /// May answer with `{"notice": "..."}`, which is appended to the result the agent
    /// reads -- the one place it is certainly looking. Anything else on stdout, including
    /// nothing, adds nothing. A non-zero exit is logged and the run continues: an
    /// after-hook reports, and a report that fails must not fail the work it describes.
    ///
    /// The point of the notice is that it arrives while the agent can still act. A
    /// condition discovered after the run -- when the workspace is saved, say -- is
    /// discovered too late to be fixed by the run that caused it.
    pub after_tool: Vec<String>,
}

/// A fully resolved tool server definition.
///
/// Parsing from YAML and loading from disk is handled by
/// `crate::infra::persistence::tool_def`.
///
/// # Where a server is
///
/// Three arrangements, and the fields say which:
///
/// | `command` | `url` | what it means |
/// |---|---|---|
/// | set | absent | a child process, spoken to over stdio |
/// | absent | set | something already running, spoken to over HTTP |
/// | set | set | atoma starts it, then speaks HTTP to it |
///
/// The third is not a curiosity. A server atoma starts is one whose stderr atoma
/// owns, which is what keeps `domain::tool_health`'s fallback channel working and
/// what lets a credential be routed to it -- and neither of those reaches a server
/// somebody else is running. Whether the conversation then happens over a pipe or
/// a socket is a separate question from who started it.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    /// The program to start. Empty when the server is already running.
    pub command: String,
    pub args: Vec<String>,
    /// The environment the child is started with. Empty when there is no child --
    /// which is why a remote server declaring one is refused rather than ignored:
    /// a credential you believe you routed and did not is the worst of the three
    /// outcomes.
    pub env: HashMap<String, String>,
    /// Where to reach the server over Streamable HTTP. `None` means stdio.
    pub url: Option<String>,
    /// Headers sent with every request to `url`.
    ///
    /// This is how a remote server is authenticated, and it is the whole of it.
    /// `env` cannot reach an endpoint somebody else is running, so a token gets
    /// there as a header or not at all.
    pub headers: HashMap<String, String>,
    pub hooks: Hooks,
    /// Whether this server's tools keep their own names, with no `server__` prefix.
    ///
    /// The prefix is how a tool call is routed, so switching it off moves that job to
    /// a name table built at startup -- and makes two servers able to claim one name,
    /// which `McpRegistry::from_configs` refuses. It is off by default because a
    /// server that has not asked for it cannot collide with anything.
    ///
    /// It exists because the names a model has seen most are `read`, `grep`, `bash`.
    /// `filesystem_readonly__read_text_file` is none of them, and a run measured in
    /// this project spent 37 of 63 shell calls re-implementing `read` with a line
    /// range and `grep` by hand. Mature harnesses give their own tools bare names and
    /// prefix everything plugged in; this is that arrangement, made a setting because
    /// here every tool arrives over MCP and none of them is native.
    pub unprefixed: bool,
    /// How much of one tool result from this server reaches the model, in
    /// characters. `None` means the client's default.
    ///
    /// Per server for the same reason the timeout is: a shell server returning a
    /// test suite's output and a filesystem server returning a config file are not
    /// the same question. See `domain::tool_output` for what the default is and
    /// why a cap exists at all.
    pub max_output_chars: Option<usize>,
    /// How long one `tools/list` or `tools/call` on this server may take, in
    /// seconds. `None` means the client's default.
    ///
    /// Per server because the right value is a property of what the server does,
    /// and a single number cannot be right for all of them. A `github` server that
    /// has not answered in a minute has stopped answering. A `shell` server that
    /// has not answered in a minute is compiling -- and its own `shell_execute`
    /// advertises `timeout_seconds` up to 3600, which was unreachable while one
    /// constant capped every server at 60.
    ///
    /// Raising it is not free: this is the only thing that notices a server which
    /// has stopped responding, so a large value means a long wait before a stuck
    /// run says so. Which is why it is opt-in per server rather than a bigger
    /// default, and why a server that answers quickly should not set it.
    pub request_timeout_secs: Option<u64>,
    /// Whether a line this server writes to its log channel may be given a severity
    /// guessed from the words in it.
    ///
    /// Off by default, and that is the point. `domain::tool_health` reads `error`,
    /// `fatal`, `panic`, `warn` and their plurals out of a line and attaches what it
    /// finds to the server's next tool result. The word list was calibrated on the
    /// output of servers whoever wrote it had read -- but atoma ships no MCP server,
    /// so every server is somebody else's, and a caller who only wires one up cannot
    /// correct a misreading of output they do not control.
    ///
    /// The measured case: `npx -y @modelcontextprotocol/server-filesystem` prints
    /// npm's deprecation notice before the server starts, and the run's first tool
    /// result said the `filesystem` server had reported a problem. It had not; npm
    /// had. A build tool ending with "0 errors" fails the same way.
    ///
    /// So the guess is per server, set by whoever knows that server's output -- the
    /// same layer as `request_timeout_secs` and `max_output_chars`. Off does not mean
    /// silent: the line is still logged, it just stops being presented to the model
    /// as the server's own report.
    ///
    /// `notifications/message` is unaffected. There the severity is a field the server
    /// filled in, which is a report rather than a guess, and it stays on by default.
    pub guess_severity_from_output: bool,
}

/// What to say when `mcp_servers` names a server the tools file has not got.
///
/// Lists the ones it has. The names are right there in the map both callers already
/// hold, and withholding them leaves a person to guess at a spelling -- or to go and
/// read the tools file, which under Atoma Autonomous Delivery is generated per run
/// into a temp directory and is not somewhere they can look.
///
/// This is the treatment `application::tools::unknown_skill_message` already gives an
/// unknown skill, and `validator` gives an unknown provider, for the same reason each
/// states: repeating the names at the moment of the mistake costs a line and removes
/// the guess. The MCP-server check is the one that never got it.
///
/// Sorted, because a `HashMap` iterates in whatever order it likes and an error
/// message that reshuffles itself between runs reads as a different error.
pub fn unknown_server_message<'a>(asked: &str, available: impl Iterator<Item = &'a str>) -> String {
    let mut names: Vec<&str> = available.collect();
    names.sort_unstable();
    if names.is_empty() {
        return format!(
            "mcp_servers '{}': the tools file declares no servers at all.",
            asked
        );
    }
    format!(
        "mcp_servers '{}': no server by that name. The tools file declares: {}.",
        asked,
        names.join(", ")
    )
}

#[cfg(test)]
mod unknown_server_message_tests {
    use super::unknown_server_message;

    #[test]
    fn it_names_what_exists() {
        let have = ["shell", "github"];
        let message = unknown_server_message("githbu", have.iter().copied());
        assert!(
            message.contains("githbu"),
            "it says what was asked for: {message}"
        );
        assert!(message.contains("github"), "and what exists: {message}");
    }

    /// A `HashMap` iterates in whatever order it likes. Two runs of the same mistake
    /// producing two different messages is how a reader concludes something changed.
    #[test]
    fn the_order_does_not_depend_on_the_map() {
        let one = unknown_server_message("x", ["web", "atoma", "shell"].iter().copied());
        let two = unknown_server_message("x", ["shell", "web", "atoma"].iter().copied());
        assert_eq!(one, two);
        assert!(one.contains("atoma, shell, web"), "sorted: {one}");
    }

    /// An empty tools file is a different mistake, and "the ones that exist are: ."
    /// is not a sentence.
    #[test]
    fn an_empty_tools_file_says_so() {
        let message = unknown_server_message("shell", std::iter::empty());
        assert!(message.contains("no servers at all"), "{message}");
    }
}
