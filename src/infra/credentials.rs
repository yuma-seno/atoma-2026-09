//! Where credential values come from, and the one place that decides.
//!
//! # Two sources, never both
//!
//! A file, or the environment. If `--credentials-file` is given, that file is the
//! only source and the environment is not consulted for credentials at all; if it
//! is not, values are read from the environment exactly as they always were.
//!
//! Two sources rather than one because the contexts genuinely differ. A file
//! delivered by CI is ephemeral and can be deleted the moment it has been read; a
//! developer's is a config file that belongs to them and must not be. Collapsing
//! them into one concept would mean a flag deciding whether to delete, and would
//! push a hand-run `atoma` into keeping credentials in plaintext on disk. The
//! environment is the better answer there.
//!
//! # Why a file at all
//!
//! Because a value in an environment block cannot be taken back. `/proc/<pid>/
//! environ` reflects what was placed on the stack at `execve`, and glibc's
//! `setenv`/`unsetenv` do not rewrite it — so anything that was ever in a
//! process's environment stays readable there, by any process of the same user,
//! for that process's lifetime. Measured, not assumed.
//!
//! A file has the one property the environment lacks: its exposure can be ended.
//! So `from_file` reads it and immediately deletes it, and the caller does that
//! before any tool server exists. The file and the servers never coexist.
//!
//! This is the arrangement Kubernetes settled on for the same reasons — secrets
//! mounted as files rather than injected as environment variables — so it is the
//! ordinary shape rather than a workaround.
//!
//! # What this does NOT do
//!
//! It does not put anything into this process's own environment. That would put
//! it back in `/proc` and undo the point. Values live in this map, and reach a
//! child only by being named in that child's `env` in the tools file — see
//! `expand`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};

/// The names the caller declared, or unset when nothing has declared any.
///
/// Process-wide rather than carried through `McpFactory`, because two separate
/// paths start a tool server and only one of them goes through a factory: a run,
/// via `McpRegistry::from_configs`, and `atoma validate --with-live-tools`, via
/// `infra::mcp::inspect` -- a free function that takes the definitions and nothing
/// else. Threading the list through the first would have left the second starting
/// `shell` with the caller's own secrets still in its environment, and nothing
/// would have failed or said so. Registered once, before either path can exist, it
/// covers both by construction.
static DECLARED_ENV_NAMES: OnceLock<Vec<String>> = OnceLock::new();

/// Record the names the caller asked to keep out of every tool server.
///
/// This is the whole of the caller's half, and it is values rather than knowledge:
/// GitLab's `CI_JOB_TOKEN`, a Slack token, an in-house secret named after an
/// in-house system. atoma cannot know them, and a core that tried to guess would be
/// guessing forever, so `protect_env = [...]` in `atoma.toml` is how a project says
/// which of its own secrets a `shell` server must not inherit.
///
/// A union with the provider names rather than a replacement for them: declaring
/// one name must not be a way to un-protect a provider key.
///
/// Must be called before any tool server is started, and `main` does it as soon as
/// the configuration is known. A second call is a defect rather than an override --
/// two answers to "what is secret here" means one of them is being ignored -- so the
/// first stands, because it is the one any server already started was built against.
pub fn declare_protected_env_names(names: Vec<String>) {
    if DECLARED_ENV_NAMES.set(sanitised(names)).is_err() {
        tracing::warn!(
            "protected environment names were declared more than once; the first declaration stands and the later one is ignored"
        );
    }
}

/// Drop what an environment variable name cannot be.
///
/// A trailing space inside a TOML string is easy to write and impossible to see,
/// and `env_remove("GH_TOKEN ")` removes nothing at all. An entry that silently
/// protects nothing is the worst kind of entry for a protection list to carry,
/// because the list reads as though it covered the name.
fn sanitised(names: Vec<String>) -> Vec<String> {
    names
        .into_iter()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect()
}

/// The environment variables this program treats as credentials.
///
/// The provider keys `infra/llm/*` declares, plus whatever the caller declared with
/// [`declare_protected_env_names`]. Not an attempt to enumerate every secret a
/// variable could hold; a list like that is wrong the moment someone invents a name
/// nobody here thought of, which is exactly why the second half is taken from the
/// caller instead of guessed.
///
/// Used to keep them out of the environment of the tool servers this process
/// spawns. A server that legitimately needs one names it in its own `env` in the
/// tools file, which is applied after the removal and so puts it back — for that
/// one server and no other.
///
/// In file mode this is mostly moot, because none of these are in this process's
/// environment to begin with. It earns its place in environment mode, which is
/// how a developer runs atoma by hand: there the provider key really is inherited
/// by every server, and `shell` could read it out of its own environment without
/// going anywhere near `/proc`.
///
/// Nothing is written out by hand here any more. This used to open with three
/// GitHub names that were, each for a different reason, not knowledge this crate
/// held. `GH_TOKEN` and `GITHUB_TOKEN` are declared by the Copilot provider and so
/// are already in the provider half -- `provider_credential_names` is built from the
/// whole static `PROVIDERS` table rather than from the provider a run resolved to,
/// so they are stripped whichever provider is in use, and the test below fails if
/// that ever stops being true. `GITHUB_PERSONAL_ACCESS_TOKEN` was read by no code in
/// this repository and by none in the embedder that asked for it, which has since
/// been retired: a name that entered the core for one caller's convenience and
/// outlived the convenience.
pub fn credential_env_names() -> Vec<String> {
    match DECLARED_ENV_NAMES.get() {
        Some(declared) => protected_union(declared),
        None => protected_union(&[]),
    }
}

/// The union itself, over a declared list given rather than one registered.
///
/// A union rather than a list: the provider half is whatever `infra::llm` declares,
/// so adding a provider covers it here with nothing to remember, and the caller's
/// half is whatever the caller said.
///
/// Split out from `credential_env_names` so the rule can be tested at all.
/// `declare_protected_env_names` writes a `OnceLock` the whole test binary shares,
/// so a test that called it would decide the answer for every test that ran after
/// it -- and which tests those are depends on the order the harness happens to pick.
fn protected_union(declared: &[String]) -> Vec<String> {
    let mut names: Vec<String> = crate::infra::llm::provider_credential_names()
        .into_iter()
        .map(String::from)
        .collect();
    names.extend_from_slice(declared);
    names.sort_unstable();
    names.dedup();
    names
}

/// The credential values available to this run.
pub struct Credentials {
    /// `Some` when a file was supplied, and then the only source. `None` means
    /// read from the environment.
    values: Option<HashMap<String, String>>,
}

impl Credentials {
    /// Read a JSON object of `{"NAME": "value"}` and delete the file.
    ///
    /// Deleting here rather than leaving it to the caller is deliberate: the
    /// guarantee is that no tool server ever coexists with the file, and the only
    /// way to be sure is for the read and the delete to be the same act. A
    /// workflow that removed it afterwards would leave it readable for the whole
    /// run.
    ///
    /// A file that cannot be deleted is a warning rather than a failure. The
    /// values are already in memory, the run can proceed, and saying so is more
    /// use than refusing to start.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read credentials file: {:?}", path))?;

        let values: HashMap<String, String> =
            serde_json::from_str(&content).with_context(|| {
                format!(
                    "Credentials file is not a JSON object of name/value pairs: {:?}",
                    path
                )
            })?;

        if let Err(error) = fs::remove_file(path) {
            tracing::warn!(
                ?path,
                %error,
                "could not delete the credentials file after reading it; it stays readable to anything running as this user for the rest of the run"
            );
        }

        tracing::debug!(
            count = values.len(),
            "credentials read from file and the file removed"
        );
        Ok(Self {
            values: Some(values),
        })
    }

    /// Read credentials from the environment, as before this existed.
    pub fn from_environment() -> Self {
        Self { values: None }
    }

    /// The value for `name`, from whichever source this was built with.
    pub fn get(&self, name: &str) -> Option<String> {
        match &self.values {
            Some(values) => values.get(name).cloned(),
            None => std::env::var(name).ok(),
        }
    }

    /// Whether `name` has a non-empty value.
    ///
    /// Used to decide which provider a run is for, so an empty string has to
    /// count as absent: a workflow that exports a secret which is not set passes
    /// one, and auto-detection reading that as "OpenAI is configured" is how a
    /// run ends up failing with the wrong provider's error message.
    pub fn has(&self, name: &str) -> bool {
        self.get(name).is_some_and(|value| !value.is_empty())
    }

    /// Replace `${NAME}` in `template` with the credential of that name.
    ///
    /// This is how a value reaches one tool server and not the others: the tools
    /// file names it in that server's `env`, and nothing else sees it. An unknown
    /// name expands to empty with a warning rather than failing the run — a
    /// project that has declared a credential it has not added yet should get a
    /// server that cannot authenticate, and a log line saying why, not a run that
    /// refuses to start.
    pub fn expand(&self, template: &str) -> String {
        expand_with(template, |name| self.get(name), "credential")
    }
}

/// Substitute `${NAME}` and `${NAME:-default}` from whatever `lookup` returns.
///
/// Shared by two callers that must NOT share a source. A credential belongs only
/// in the environment of the server that declared it, so `env:` values resolve
/// against the credentials; a program path is not a secret and must work when no
/// credentials exist at all, so `args` resolve against the process environment.
/// Keeping the substitution common and the sources separate is what stops a path
/// from becoming a way to read a credential.
///
/// An unknown name expands to its default, or to empty with a warning when it has
/// none. Failing the run instead would turn a project declaring a credential it
/// has not added yet into a repository whose agents cannot start.
fn expand_with(template: &str, lookup: impl Fn(&str) -> Option<String>, what: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;

    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find('}') else {
            // No closing brace: not a reference, so it is literal text.
            out.push_str(&rest[start..]);
            return out;
        };

        let (name, fallback) = match after[..end].split_once(":-") {
            Some((name, fallback)) => (name, Some(fallback)),
            None => (&after[..end], None),
        };

        match lookup(name) {
            Some(value) => out.push_str(&value),
            None => match fallback {
                Some(fallback) => out.push_str(fallback),
                None => tracing::warn!(
                    name,
                    what,
                    "a tools file references something that is not available; it will be empty"
                ),
            },
        }
        rest = &after[end + 1..];
    }

    out.push_str(rest);
    out
}

/// Substitute `${NAME}` in a tools file's `args` from the process environment.
///
/// Deliberately not the credentials. These are program paths -- which is how the
/// delivery runner points a tool server at a checkout of the default branch
/// rather than at the pull request under review -- and a path that only resolved
/// when a credential of the same name existed would be a trap.
pub fn expand_from_environment(template: &str) -> String {
    expand_with(
        template,
        |name| std::env::var(name).ok(),
        "environment variable",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// The GitHub tokens used to be written out here in a list of their own, and
    /// deleting that list is only safe because `provider_credential_names` is built
    /// from the whole static `PROVIDERS` table rather than from the provider a run
    /// resolved to -- Copilot declares both, so both are stripped from every tool
    /// server whichever provider the run is using.
    ///
    /// If that ever stops being true -- Copilot removed, or its `credential_names`
    /// narrowed -- this fails here, at the place that decides, rather than silently
    /// in a `shell` server's environment where nothing would report it.
    #[test]
    fn the_github_tokens_are_protected_without_a_list_of_their_own() {
        let names = protected_union(&[]);
        for name in ["GH_TOKEN", "GITHUB_TOKEN"] {
            assert!(
                names.iter().any(|n| n == name),
                "{name} is missing from {names:?}"
            );
        }
    }

    /// A caller's own secret cannot be guessed by this crate, so it arrives as a
    /// value. GitLab's `CI_JOB_TOKEN` was the measured case: inherited straight into
    /// `shell` because nothing here had ever heard of it.
    ///
    /// A union, never a replacement: declaring one name must not be a way to make a
    /// provider key inheritable again.
    #[test]
    fn a_declared_name_joins_the_provider_names_rather_than_replacing_them() {
        let names = protected_union(&["CI_JOB_TOKEN".to_string()]);
        let has = |name: &str| names.iter().any(|n| n == name);
        assert!(has("CI_JOB_TOKEN"), "{names:?}");
        assert!(has("ANTHROPIC_API_KEY"), "{names:?}");
    }

    /// Declaring a name the provider half already carries is not an error, and the
    /// answer is still a set. `env_remove` is idempotent, so a duplicate would do no
    /// harm at the call site -- but a list that can contain a name twice is not the
    /// union this claims to be, and the next reader would have to work that out.
    #[test]
    fn a_name_declared_twice_over_appears_once() {
        let declared = vec!["GH_TOKEN".to_string(), "GH_TOKEN".to_string()];
        let names = protected_union(&declared);
        let count = names.iter().filter(|n| *n == "GH_TOKEN").count();
        assert_eq!(count, 1, "{names:?}");
    }

    /// A trailing space inside a TOML string is invisible to whoever wrote it, and
    /// `env_remove("CI_JOB_TOKEN ")` removes nothing. A protection list that reads as
    /// though it covered a name and does not is the one failure it cannot afford.
    #[test]
    fn a_padded_or_empty_declaration_is_dropped_rather_than_registered() {
        let given = vec![
            " CI_JOB_TOKEN ".to_string(),
            "".to_string(),
            "   ".to_string(),
        ];
        assert_eq!(sanitised(given), vec!["CI_JOB_TOKEN".to_string()]);
    }

    fn from_pairs(pairs: &[(&str, &str)]) -> Credentials {
        Credentials {
            values: Some(
                pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                    .collect(),
            ),
        }
    }

    #[test]
    fn a_file_is_read_and_then_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("creds.json");
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(br#"{"OPENAI_API_KEY":"sk-test","SLACK_TOKEN":"xoxb-test"}"#)
            .unwrap();
        drop(file);

        let credentials = Credentials::from_file(&path).unwrap();

        assert_eq!(
            credentials.get("OPENAI_API_KEY"),
            Some("sk-test".to_string())
        );
        assert!(
            !path.exists(),
            "the file must be gone before any tool server can read it"
        );
    }

    /// The ordering the guarantee rests on, pinned against a refactor.
    ///
    /// `from_file` deletes as it reads, so "the file is gone before any tool
    /// server starts" holds only while the read happens before the command runs.
    /// Move it after `match cli.command` and the file would sit on disk for the
    /// whole run, readable by every server the agent spawns -- and nothing would
    /// fail. The guarantee would simply be gone.
    ///
    /// Checking the source is crude, and it is the only thing that catches this:
    /// a behavioural test would need a full agent run, and the failure it guards
    /// against is invisible at runtime.
    #[test]
    fn credentials_are_read_before_the_command_runs() {
        let main_rs = include_str!("../main.rs");
        let read_at = main_rs
            .find("Credentials::from_file")
            .expect("main.rs must build credentials from a file");
        let dispatch_at = main_rs
            .find("match cli.command")
            .expect("main.rs must dispatch on the subcommand");
        assert!(
            read_at < dispatch_at,
            "credentials must be read (and the file deleted) before any command runs, or the file outlives the tool servers"
        );
    }

    #[test]
    fn a_malformed_file_fails_rather_than_yielding_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("creds.json");
        fs::write(&path, "not json").unwrap();
        assert!(Credentials::from_file(&path).is_err());
    }

    /// File mode means the file is the ONLY source. Falling through to the
    /// environment would reintroduce exactly what the file exists to avoid.
    #[test]
    fn file_mode_does_not_fall_back_to_the_environment() {
        // SAFETY: single-threaded test, and the name is unique to it.
        unsafe { std::env::set_var("ATOMA_TEST_ONLY_IN_ENV", "from-env") };
        let credentials = from_pairs(&[("OTHER", "x")]);
        assert_eq!(credentials.get("ATOMA_TEST_ONLY_IN_ENV"), None);
        unsafe { std::env::remove_var("ATOMA_TEST_ONLY_IN_ENV") };
    }

    #[test]
    fn environment_mode_reads_the_environment() {
        // SAFETY: as above.
        unsafe { std::env::set_var("ATOMA_TEST_ENV_MODE", "yes") };
        let credentials = Credentials::from_environment();
        assert_eq!(
            credentials.get("ATOMA_TEST_ENV_MODE"),
            Some("yes".to_string())
        );
        unsafe { std::env::remove_var("ATOMA_TEST_ENV_MODE") };
    }

    /// An exported-but-unset secret arrives as an empty string. Reading that as
    /// "configured" is how a run picks the wrong provider and then fails with a
    /// message about the one it did not want.
    #[test]
    fn an_empty_value_counts_as_absent() {
        let credentials = from_pairs(&[("OPENAI_API_KEY", "")]);
        assert!(!credentials.has("OPENAI_API_KEY"));
        assert!(from_pairs(&[("OPENAI_API_KEY", "sk-x")]).has("OPENAI_API_KEY"));
    }

    #[test]
    fn expand_substitutes_a_reference() {
        let credentials = from_pairs(&[("SLACK_TOKEN", "xoxb-1")]);
        assert_eq!(credentials.expand("${SLACK_TOKEN}"), "xoxb-1");
        assert_eq!(
            credentials.expand("Bearer ${SLACK_TOKEN} end"),
            "Bearer xoxb-1 end"
        );
    }

    #[test]
    fn expand_leaves_text_without_a_reference_alone() {
        let credentials = from_pairs(&[("A", "1")]);
        assert_eq!(credentials.expand("plain value"), "plain value");
        assert_eq!(credentials.expand(""), "");
        // Unterminated: literal, not a reference.
        assert_eq!(credentials.expand("${UNCLOSED"), "${UNCLOSED");
    }

    #[test]
    fn expand_yields_empty_for_a_name_it_does_not_have() {
        let credentials = from_pairs(&[("A", "1")]);
        assert_eq!(credentials.expand("x${MISSING}y"), "xy");
    }

    /// A tools file has to keep working where the variable is unset -- a hand-run
    /// `atoma` sets no machinery root, and a program path of `/…` would not exist.
    #[test]
    fn a_default_covers_an_absent_name() {
        let credentials = from_pairs(&[("A", "1")]);
        assert_eq!(credentials.expand("${MISSING:-fallback}"), "fallback");
        assert_eq!(credentials.expand("${A:-fallback}"), "1");
        assert_eq!(credentials.expand("${MISSING:-}/x"), "/x");
    }

    /// The split that keeps a program path from becoming a way to read a secret.
    #[test]
    fn args_expansion_reads_the_environment_and_not_the_credentials() {
        // SAFETY: single-threaded test, and the names are unique to it.
        unsafe { std::env::set_var("ATOMA_TEST_ARG_ROOT", "machinery") };
        assert_eq!(
            expand_from_environment("${ATOMA_TEST_ARG_ROOT}/x.ts"),
            "machinery/x.ts"
        );
        assert_eq!(
            expand_from_environment("${ATOMA_TEST_ARG_ROOT_UNSET:-.}/x.ts"),
            "./x.ts"
        );
        unsafe { std::env::remove_var("ATOMA_TEST_ARG_ROOT") };
    }

    #[test]
    fn expand_handles_several_references() {
        let credentials = from_pairs(&[("A", "1"), ("B", "2")]);
        assert_eq!(credentials.expand("${A}-${B}-${A}"), "1-2-1");
    }
}
