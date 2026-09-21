use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::domain::tool::{Hooks, ToolDef};
use crate::infra::credentials::{expand_from_environment, Credentials};

/// YAML deserialization view — private to this module.
#[derive(Deserialize)]
struct ToolConfig {
    /// Optional now, because a server that is already running has no command.
    /// Which of `command` and `url` are present is checked in [`transport_of`],
    /// where the error can say what the four combinations mean.
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Where the server is, for one that is reached over Streamable HTTP.
    #[serde(default)]
    pub url: Option<String>,
    /// Headers for every request to `url`.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub hooks: HooksConfig,
    /// Name this server's tools as they are, with no `server__` prefix.
    ///
    /// One meaning only: what the model sees. See `domain::tool::ToolDef`.
    #[serde(default)]
    pub unprefixed: bool,
    /// Seconds one `tools/list` or `tools/call` on this server may take. Absent
    /// means the client's default, which is what nearly every server wants.
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
    /// Characters of one tool result that reach the model. Absent means the
    /// client's default.
    #[serde(default)]
    pub max_output_chars: Option<usize>,
    /// Let a severity be guessed from the words in this server's output.
    ///
    /// Absent means no, which is the only defensible default for a word list
    /// calibrated on servers the caller may not be running. See
    /// `domain::tool::ToolDef` for the npm deprecation notice that got attached to a
    /// tool result as a problem the `filesystem` server had reported.
    #[serde(default)]
    pub guess_severity_from_output: bool,
}

/// The whole tools file: one reserved key, and a server under every other.
///
/// `hooks` at the top level applies to every server. It exists because the thing a
/// repository most often wants to watch is not a tool but the run -- how much has been
/// written where, how long a search has gone on -- and attaching that to one server
/// only watches the agent while it happens to be using that server.
///
/// `#[serde(flatten)]` over a map is what makes `hooks` reserved: it is taken first, and
/// every remaining key is a server. A server actually named `hooks` is the cost, and it
/// fails loudly at load rather than quietly at run time.
#[derive(Deserialize)]
struct ToolsFile {
    #[serde(default)]
    hooks: HooksConfig,
    #[serde(flatten)]
    servers: HashMap<String, ToolConfig>,
}

/// One hook script, or several.
///
/// `Hooks` has always held a `Vec`, and the loader has always concatenated the
/// file-wide scripts with each server's. Only this file's shape was singular, which
/// meant a tools file could declare at most one hook per level even though
/// everything downstream was built to run a list.
///
/// That narrowing is felt by whoever generates the file. A delivery that ships a
/// required hook of its own and wants to let a project add theirs had, with one slot,
/// no way to express both -- and the workaround is a dispatcher script that
/// re-implements ordering, spawning and error handling the loader already does.
///
/// Both spellings, because `after_tool: ./guard.ts` is the common case and reads
/// better than a one-element list.
#[derive(Deserialize)]
#[serde(untagged)]
enum HookScripts {
    One(String),
    Many(Vec<String>),
}

impl HookScripts {
    fn into_vec(self) -> Vec<String> {
        match self {
            HookScripts::One(script) => vec![script],
            HookScripts::Many(scripts) => scripts,
        }
    }
}

#[derive(Deserialize, Default)]
struct HooksConfig {
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    #[serde(default)]
    pub tool_denylist: Vec<String>,
    #[serde(default)]
    pub before_tool: Option<HookScripts>,
    #[serde(default)]
    pub after_tool: Option<HookScripts>,
}

/// Which of the four combinations of `command` and `url` this entry is, or an
/// error naming what is missing.
///
/// A function so the reasoning sits next to the rule. Three combinations are
/// meaningful and one is not:
///
///   - `command`          -- a child, over stdio
///   - `url`              -- something already running, over HTTP
///   - `command` + `url`  -- atoma starts it, then speaks HTTP to it
///   - neither            -- nothing to talk to
///
/// The last used to be impossible to write, because `command` was required. Now it
/// is possible and is refused here, before anything starts: a tools file naming a
/// server with no way to reach it would otherwise fail at connection time with an
/// error about an empty program name.
fn transport_of(name: &str, cfg: &ToolConfig) -> Result<()> {
    let has_command = !cfg.command.trim().is_empty();
    let url = cfg.url.as_deref().map(str::trim).unwrap_or("");
    let has_url = !url.is_empty();

    if !has_command && !has_url {
        anyhow::bail!(
            "Tool server '{}' names neither 'command' nor 'url', so there is nothing to \
             connect to. Give it a 'command' to start a server over stdio, a 'url' to \
             reach one that is already running, or both to start one and reach it over HTTP.",
            name,
        );
    }

    if has_url && !url.starts_with("http://") && !url.starts_with("https://") {
        anyhow::bail!(
            "Tool server '{}' has url '{}', which is not an http:// or https:// address. \
             Streamable HTTP is the only transport a url names.",
            name,
            url,
        );
    }

    // Refused rather than ignored, and this is the case worth being strict about.
    // `env` is how a credential reaches a server, and it reaches it by being placed
    // in a child process's environment. There is no child here, so the value goes
    // nowhere -- and a credential someone believes they routed and did not is worse
    // than either a failure or an honest absence. A remote endpoint is
    // authenticated by `headers`.
    if has_url && !has_command && !cfg.env.is_empty() {
        anyhow::bail!(
            "Tool server '{}' declares 'env' but no 'command', so nothing is started and \
             those values reach nothing. A server at a url is authenticated with 'headers'.",
            name,
        );
    }

    // The mirror of the above, for the same reason: a header on a server nobody
    // sends a request to is a token that was never sent.
    if !has_url && !cfg.headers.is_empty() {
        anyhow::bail!(
            "Tool server '{}' declares 'headers' but no 'url'. A server spoken to over stdio \
             has no requests to put them on; a credential reaches it through 'env'.",
            name,
        );
    }

    Ok(())
}

/// Load a tools YAML file and return a map of server-name → `ToolDef`.
///
/// Hook script paths are resolved relative to the directory of the YAML file,
/// so `./scripts/guard.py` in `./tools/tools.yaml` resolves to
/// `./tools/scripts/guard.py` regardless of the working directory.
///
/// # Example YAML
/// ```yaml
/// filesystem:
///   command: npx
///   args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
///   hooks:
///     tool_allowlist: ["filesystem__*"]
///     before_tool: ./scripts/fs_guard.py
///
/// shell:
///   command: bun
///   args: ["run", "./scripts/shell.ts"]
///   # A build is not a stall. Only set this for a server whose work genuinely
///   # takes minutes -- it is also the only thing that notices a hung server.
///   request_timeout_secs: 3600
///
/// # Already running somewhere else. No process, so no `env`: the token is a
/// # header, because that is the only thing that reaches an endpoint atoma did
/// # not start.
/// warehouse:
///   url: https://mcp.internal.example.com/mcp
///   headers:
///     Authorization: "Bearer ${WAREHOUSE_TOKEN}"
/// ```
pub fn load(path: &Path, credentials: &Credentials) -> Result<HashMap<String, ToolDef>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read tools file: {:?}", path))?;
    let file: ToolsFile = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse tools YAML: {:?}", path))?;

    let base_dir = path.parent().unwrap_or(Path::new("."));

    // Every script the entry names, in the order it named them. Order is the whole
    // contract for `before_tool` -- the first refusal wins and the rest do not run --
    // so it is preserved rather than sorted or deduplicated.
    let resolve = |s: Option<HookScripts>| -> Result<Vec<String>> {
        let Some(scripts) = s else {
            return Ok(Vec::new());
        };
        scripts
            .into_vec()
            .into_iter()
            .map(|script| {
                let p = Path::new(&script);
                let resolved = if p.is_absolute() {
                    script.clone()
                } else {
                    base_dir.join(p).to_string_lossy().into_owned()
                };
                if !Path::new(&resolved).exists() {
                    anyhow::bail!(
                        "Hook script not found: '{}' (resolved from '{}')",
                        resolved,
                        script
                    );
                }
                Ok(resolved)
            })
            .collect()
    };

    // Resolved once, then cloned onto every server: the paths are the same paths, and
    // a missing file should be reported once rather than once per server.
    let file_wide_before = resolve(file.hooks.before_tool)?;
    let file_wide_after = resolve(file.hooks.after_tool)?;

    file.servers
        .into_iter()
        .map(|(name, cfg)| {
            transport_of(&name, &cfg)?;
            // File-wide first, so a repository-wide rule cannot be skipped by a server
            // hook that refuses before it, and a repository-wide notice is read before
            // whatever the server itself has to add.
            let hooks = Hooks {
                tool_allowlist: cfg.hooks.tool_allowlist,
                tool_denylist: cfg.hooks.tool_denylist,
                before_tool: file_wide_before
                    .iter()
                    .cloned()
                    .chain(resolve(cfg.hooks.before_tool)?)
                    .collect(),
                after_tool: file_wide_after
                    .iter()
                    .cloned()
                    .chain(resolve(cfg.hooks.after_tool)?)
                    .collect(),
            };
            let def = ToolDef {
                name: name.clone(),
                unprefixed: cfg.unprefixed,
                command: cfg.command,
                // `${NAME}` here resolves against the ENVIRONMENT, not the
                // credentials. These are program paths: the delivery runner uses
                // one to point a tool server at a checkout of the default branch
                // rather than at the pull request under review, so a path that
                // only resolved when a credential of that name existed would be a
                // trap. `${NAME:-default}` keeps a tools file working where the
                // variable is unset, such as a hand-run `atoma`.
                args: cfg
                    .args
                    .into_iter()
                    .map(|arg| expand_from_environment(&arg))
                    .collect(),
                // `${NAME}` resolved here, against the run's credentials rather
                // than the environment. This is what routes a credential to one
                // server and not the others: a value reaches a tool only by being
                // named in that tool's `env`.
                //
                // Values were literal before this, so a tools file written for an
                // older atoma is unaffected -- there is nothing to expand in it.
                env: cfg
                    .env
                    .into_iter()
                    .map(|(key, value)| (key, credentials.expand(&value)))
                    .collect(),
                // The environment, not the credentials -- a url is an address and
                // the delivery runner already points server paths at a checkout
                // the same way. Same reasoning as `args` above.
                url: cfg
                    .url
                    .map(|url| expand_from_environment(url.trim()))
                    .filter(|url| !url.is_empty()),
                // The credentials, like `env` -- these carry the token. Same
                // routing rule: a value reaches a server only by being named in
                // that server's own block.
                headers: cfg
                    .headers
                    .into_iter()
                    .map(|(key, value)| (key, credentials.expand(&value)))
                    .collect(),
                hooks,
                // Zero means the default, the same as absent. `infra::timeouts`
                // made that the rule for every timeout read from the environment
                // after three of the four call sites took `0` literally and turned
                // a stall detector into an immediate failure. A tools file is a
                // different source, but the reader is the same person, and a rule
                // that holds in one place and not the other is worse than either.
                request_timeout_secs: cfg.request_timeout_secs.filter(|secs| *secs > 0),
                // Zero means the default here too. One rule about zero across the
                // whole crate is worth more than a cleverer rule in one place.
                max_output_chars: cfg.max_output_chars.filter(|chars| *chars > 0),
                // Straight through: a bool has no zero to reinterpret, and the
                // absent case is already false by `#[serde(default)]`.
                guess_severity_from_output: cfg.guess_severity_from_output,
            };
            Ok((name, def))
        })
        .collect()
}

// ── Port adapter ──────────────────────────────────────────────────────────────

/// File-system adapter implementing `ToolDefPort`.
///
/// Holds the run's credentials rather than taking them per call, because the
/// port's `load` is what the runner sees and adding a parameter there would push
/// credentials through every caller that has no business with them.
pub struct FileToolDefAdapter {
    credentials: Credentials,
}

impl FileToolDefAdapter {
    pub fn new(credentials: Credentials) -> Self {
        Self { credentials }
    }
}

impl Default for FileToolDefAdapter {
    /// For `atoma validate` and for tests, which check a tools file's shape and
    /// have no run to draw credentials from. Reading the environment is what
    /// happened before credentials existed, and expanding a reference that is not
    /// set yields empty, which validation does not care about.
    fn default() -> Self {
        Self::new(Credentials::from_environment())
    }
}

impl crate::domain::ports::ToolDefPort for FileToolDefAdapter {
    fn load(
        &self,
        path: &std::path::Path,
    ) -> anyhow::Result<std::collections::HashMap<String, crate::domain::tool::ToolDef>> {
        load(path, &self.credentials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn load_yaml(body: &str) -> HashMap<String, ToolDef> {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(body.as_bytes()).expect("write");
        load(file.path(), &Credentials::from_environment()).expect("load")
    }

    /// A hook script file, and the path to it as YAML will carry it.
    ///
    /// Single-quoted, because a Windows path holds backslashes and a plain scalar
    /// would leave them for the parser to interpret.
    fn hook_in(dir: &std::path::Path, name: &str) -> (String, String) {
        let path = dir.join(name);
        std::fs::write(&path, "").expect("write hook");
        let path = path.to_string_lossy().into_owned();
        let quoted = format!("'{}'", path);
        (path, quoted)
    }

    /// The narrowing this widened. `Hooks` has always held a `Vec`, and the loader has
    /// always concatenated the file-wide scripts with each server own list; only this
    /// file shape was singular. A delivery that ships a required hook AND lets a project
    /// add one had, with a single slot, no way to say both.
    #[test]
    fn a_list_of_hooks_is_kept_in_the_order_it_was_written() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (first, first_yaml) = hook_in(dir.path(), "first.ts");
        let (second, second_yaml) = hook_in(dir.path(), "second.ts");
        let body = format!(
            "hooks:\n  after_tool:\n    - {}\n    - {}\nshell:\n  command: bun\n  args: []\n",
            first_yaml, second_yaml
        );
        let tools = load_yaml(&body);
        assert_eq!(tools["shell"].hooks.after_tool, vec![first, second]);
    }

    /// The common case, and the spelling every existing tools file uses. One script
    /// reads better as a scalar than as a one-element list, so both are accepted.
    #[test]
    fn a_single_hook_is_still_a_bare_string() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (only, only_yaml) = hook_in(dir.path(), "only.ts");
        let body = format!(
            "hooks:\n  after_tool: {}\nshell:\n  command: bun\n  args: []\n",
            only_yaml
        );
        let tools = load_yaml(&body);
        assert_eq!(tools["shell"].hooks.after_tool, vec![only]);
    }

    /// File-wide first, then the server own -- the order the loader already promised,
    /// now that each side can be several.
    #[test]
    fn file_wide_hooks_run_before_the_servers_own() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (wide, wide_yaml) = hook_in(dir.path(), "wide.ts");
        let (mine, mine_yaml) = hook_in(dir.path(), "mine.ts");
        let body = format!(
            "hooks:\n  before_tool: {}\nshell:\n  command: bun\n  args: []\n  hooks:\n    before_tool:\n      - {}\n",
            wide_yaml, mine_yaml
        );
        let tools = load_yaml(&body);
        assert_eq!(tools["shell"].hooks.before_tool, vec![wide, mine]);
    }

    #[test]
    fn a_server_that_says_nothing_gets_the_client_default() {
        let tools = load_yaml("github:\n  command: bun\n  args: [\"run\", \"github.ts\"]\n");
        assert_eq!(tools["github"].request_timeout_secs, None);
    }

    /// The value `shell` needs: `shell_execute` advertises `timeout_seconds` up to
    /// 3600, and every value above the client's 60-second default was unreachable.
    #[test]
    fn a_declared_timeout_is_carried_through() {
        let tools = load_yaml("shell:\n  command: bun\n  args: []\n  request_timeout_secs: 3600\n");
        assert_eq!(tools["shell"].request_timeout_secs, Some(3600));
    }

    /// Same rule as `infra::timeouts`: zero means the default, not "fail every call
    /// immediately". One rule for every timeout in this codebase, because the
    /// person reading them is the same person.
    #[test]
    fn zero_means_the_default_the_same_as_absent() {
        let tools = load_yaml("web:\n  command: bun\n  args: []\n  request_timeout_secs: 0\n");
        assert_eq!(tools["web"].request_timeout_secs, None);
    }

    /// Off unless asked for, which is the whole of the change. A tools file written
    /// for an older atoma says nothing about this key, and a caller wiring up
    /// somebody else's server gets no guesses made about its output.
    #[test]
    fn a_server_that_says_nothing_gets_no_severity_guess() {
        let tools = load_yaml("shell:\n  command: bun\n  args: []\n");
        assert!(!tools["shell"].guess_severity_from_output);
    }

    /// And asking for it works, because the people who can defend the guess are the
    /// ones who have read the server's output -- the same ones who set its timeout.
    #[test]
    fn a_server_can_ask_for_the_severity_guess() {
        let tools = load_yaml("shell:\n  command: bun\n  guess_severity_from_output: true\n");
        assert!(tools["shell"].guess_severity_from_output);
    }

    fn load_err(body: &str) -> String {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(body.as_bytes()).expect("write");
        load(file.path(), &Credentials::from_environment())
            .expect_err("expected this tools file to be refused")
            .to_string()
    }

    #[test]
    fn a_command_alone_is_stdio() {
        let tools = load_yaml("shell:\n  command: bun\n  args: []\n");
        assert_eq!(tools["shell"].url, None);
        assert!(tools["shell"].headers.is_empty());
    }

    #[test]
    fn a_url_alone_needs_no_command() {
        let tools = load_yaml("warehouse:\n  url: https://mcp.example.com/mcp\n");
        assert_eq!(
            tools["warehouse"].url.as_deref(),
            Some("https://mcp.example.com/mcp"),
        );
        assert_eq!(tools["warehouse"].command, "");
    }

    /// Both, which is the arrangement that keeps a server's stderr -- and so its
    /// health reports -- while talking to it over HTTP.
    #[test]
    fn a_command_and_a_url_together_are_allowed() {
        let tools = load_yaml(
            r#"local:
  command: bun
  args: ["run", "s.ts"]
  url: http://127.0.0.1:9000/mcp
"#,
        );
        assert_eq!(tools["local"].command, "bun");
        assert!(tools["local"].url.is_some());
    }

    /// `command` used to be required, so this shape could not be written. It can
    /// now, and connecting would fail with an error about an empty program name.
    #[test]
    fn neither_a_command_nor_a_url_is_refused_at_load() {
        let message = load_err("nowhere:\n  args: []\n");
        assert!(message.contains("neither"), "{message}");
        assert!(message.contains("nowhere"), "{message}");
    }

    #[test]
    fn a_url_that_is_not_http_is_refused() {
        let message = load_err("odd:\n  url: ws://example.com/mcp\n");
        assert!(message.contains("Streamable HTTP"), "{message}");
    }

    /// The case worth being strict about: `env` puts a value in a child's
    /// environment, and there is no child. Ignoring it silently would mean a
    /// credential someone believes they routed and did not.
    #[test]
    fn env_on_a_server_with_no_process_is_refused() {
        let message = load_err(
            r#"remote:
  url: https://x.example/mcp
  env:
    GH_TOKEN: "${GH_TOKEN}"
"#,
        );
        assert!(message.contains("reach nothing"), "{message}");
        assert!(message.contains("headers"), "{message}");
    }

    /// And the mirror: a header on a server nobody sends a request to.
    #[test]
    fn headers_on_a_stdio_server_are_refused() {
        let message = load_err(
            r#"piped:
  command: bun
  headers:
    Authorization: "Bearer x"
"#,
        );
        assert!(message.contains("no 'url'"), "{message}");
        assert!(message.contains("env"), "{message}");
    }

    /// `env` is allowed the moment there is a process to put it in, even when the
    /// conversation happens over HTTP.
    #[test]
    fn env_is_allowed_when_a_command_starts_the_server() {
        let tools = load_yaml(
            r#"local:
  command: bun
  url: http://127.0.0.1:9000/mcp
  env:
    GH_TOKEN: "${GH_TOKEN}"
"#,
        );
        assert!(tools["local"].env.contains_key("GH_TOKEN"));
    }
}
