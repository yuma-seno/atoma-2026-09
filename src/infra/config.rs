/// Load and resolve Atoma configuration from atoma.toml and CLI args.
///
/// Run setting priority (highest to lowest):
///   1. CLI argument
///   2. atoma.toml profile value
///   3. atoma.toml defaults value
///   4. Hard-coded default
///
/// Environment variables use process environment, profile, then global config priority.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Top-level atoma.toml configuration file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomaConfig {
    /// Environment variable names no tool server may inherit, on top of the
    /// provider keys atoma already knows about.
    ///
    /// The caller's half of a list atoma cannot complete on its own. GitLab's
    /// `CI_JOB_TOKEN`, a Slack token, an in-house secret named after an in-house
    /// system: nothing in this crate has heard of them, so before this key they
    /// were inherited straight into every server a run started, `shell` included.
    ///
    /// Top-level rather than per-profile, and that is the point rather than an
    /// omission. A secret is a property of the project, not of the run: a name
    /// protected under `--profile review` and inherited under `--profile ship`
    /// is the kind of difference nobody notices until it has already leaked.
    /// `[env]` is per-profile because setting a variable is a choice; this is a
    /// floor.
    ///
    /// First among the fields because it is a TOML value rather than a table, and
    /// serde's TOML serializer refuses a value emitted after one.
    #[serde(default)]
    pub protect_env: Vec<String>,
    #[serde(default)]
    pub defaults: DefaultsConfig,
    #[serde(default, rename = "profile")]
    pub profiles: HashMap<String, ProfileConfig>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Default CLI argument values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DefaultsConfig {
    #[serde(default)]
    pub agent_def: Option<String>,
    #[serde(default)]
    pub tools_file: Option<String>,
    #[serde(default)]
    pub skills_dir: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub max_runtime_secs: Option<u64>,
    #[serde(default)]
    pub output: OutputFormat,
}

/// Output format for the `atoma run` command.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}

impl OutputFormat {
    /// Read the `--output` flag's value.
    ///
    /// Exists so the flag and the `output =` key cannot disagree about what the words
    /// mean. `main.rs` matched the two strings itself, in parallel with the
    /// `rename_all = "lowercase"` derive above, so adding a variant would have been
    /// accepted in a config file and refused on the command line -- one rule, expressed
    /// twice, in a parser and in its consumer.
    pub fn from_arg(raw: &str) -> anyhow::Result<Self> {
        serde_json::from_value(serde_json::Value::String(raw.to_string())).map_err(|_| {
            anyhow::anyhow!(
                "Unsupported output format: {raw}. Valid values: {}",
                Self::NAMES.join(", ")
            )
        })
    }

    /// The accepted spellings, for the message above.
    const NAMES: [&'static str; 2] = ["text", "json"];
}

/// Named profile overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileConfig {
    #[serde(default)]
    pub agent_def: Option<String>,
    #[serde(default)]
    pub tools_file: Option<String>,
    #[serde(default)]
    pub skills_dir: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub max_runtime_secs: Option<u64>,
    #[serde(default)]
    pub output: Option<OutputFormat>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Resolved configuration for a single `atoma run` invocation.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub agent_def: PathBuf,
    pub tools_file: Option<PathBuf>,
    pub skills_dir: Option<PathBuf>,
    pub template: Option<PathBuf>,
    /// The iteration ceiling, if anything asked for one.
    ///
    /// `None` is no ceiling, and that is the default. A count of iterations is a
    /// proxy for "this run has stopped getting anywhere", and a poor one in both
    /// directions: it stops a long piece of real work and lets a short useless one
    /// run. Measured in the delivery template — one task finished in 17 tool calls
    /// and the same task, framed thirteen times larger, was cut off at 200 having
    /// made 169 distinct searches and repeated only 6.
    ///
    /// What replaces it is `max_runtime`, which is not a proxy: it is the wall the
    /// job actually has.
    pub max_iterations: Option<u32>,
    /// How long the whole inference loop may run.
    ///
    /// `None` is no limit. A caller running under something that will kill it — a CI
    /// job with a timeout — should pass a value below that, so the run ends on its
    /// own terms and saves its session, instead of being killed with the session
    /// unwritten.
    pub max_runtime: Option<Duration>,
    pub output: OutputFormat,
    pub env: HashMap<String, String>,
}

/// Attempt to discover and load atoma.toml from the current directory or ancestors.
pub fn discover_and_load() -> Result<(Option<PathBuf>, Option<AtomaConfig>)> {
    let cwd = std::env::current_dir().context("Failed to get current directory")?;
    for dir in cwd.ancestors() {
        let candidate = dir.join("atoma.toml");
        if candidate.exists() {
            let content = std::fs::read_to_string(&candidate)
                .with_context(|| format!("Failed to read config: {:?}", candidate))?;
            let config: AtomaConfig = toml::from_str(&content)
                .with_context(|| format!("Failed to parse config: {:?}", candidate))?;
            tracing::info!("Loaded config from {:?}", candidate);
            return Ok((Some(candidate), Some(config)));
        }
    }
    Ok((None, None))
}

/// Resolve CLI arguments against config file.
pub struct CliOverrides {
    pub agent_def: Option<PathBuf>,
    pub tools_file: Option<PathBuf>,
    pub skills_dir: Option<PathBuf>,
    pub template: Option<PathBuf>,
    pub max_iterations: Option<u32>,
    pub max_runtime_secs: Option<u64>,
    pub output: Option<OutputFormat>,
}

pub fn resolve_run_config(
    overrides: CliOverrides,
    profile_name: Option<&str>,
    config: Option<&AtomaConfig>,
) -> Result<ResolvedConfig> {
    let (defaults, profile) =
        match config {
            Some(cfg) => {
                let prof =
                    match profile_name {
                        Some(name) => Some(cfg.profiles.get(name).cloned().with_context(|| {
                            format!("Profile '{}' not found in atoma.toml", name)
                        })?),
                        None => None,
                    };
                (Some(cfg.defaults.clone()), prof)
            }
            None if profile_name.is_some() => {
                anyhow::bail!("--profile requires an atoma.toml configuration file")
            }
            None => (None, None),
        };

    let agent_def = overrides
        .agent_def
        .clone()
        .or_else(|| {
            profile
                .as_ref()
                .and_then(|p| p.agent_def.clone())
                .map(PathBuf::from)
        })
        .or_else(|| {
            defaults
                .as_ref()
                .and_then(|d| d.agent_def.clone())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| PathBuf::from("agent.md"));

    let tools_file = overrides
        .tools_file
        .clone()
        .or_else(|| {
            profile
                .as_ref()
                .and_then(|p| p.tools_file.clone())
                .map(PathBuf::from)
        })
        .or_else(|| {
            defaults
                .as_ref()
                .and_then(|d| d.tools_file.clone())
                .map(PathBuf::from)
        });

    let skills_dir = overrides
        .skills_dir
        .or_else(|| {
            profile
                .as_ref()
                .and_then(|p| p.skills_dir.clone())
                .map(PathBuf::from)
        })
        .or_else(|| {
            defaults
                .as_ref()
                .and_then(|d| d.skills_dir.clone())
                .map(PathBuf::from)
        });

    let template = overrides
        .template
        .or_else(|| {
            profile
                .as_ref()
                .and_then(|p| p.template.clone())
                .map(PathBuf::from)
        })
        .or_else(|| {
            defaults
                .as_ref()
                .and_then(|d| d.template.clone())
                .map(PathBuf::from)
        });

    let max_iterations = overrides
        .max_iterations
        .or_else(|| profile.as_ref().and_then(|p| p.max_iterations))
        .or_else(|| defaults.as_ref().and_then(|d| d.max_iterations));

    let max_runtime = overrides
        .max_runtime_secs
        .or_else(|| profile.as_ref().and_then(|p| p.max_runtime_secs))
        .or_else(|| defaults.as_ref().and_then(|d| d.max_runtime_secs))
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs);

    let output = overrides
        .output
        .or_else(|| profile.as_ref().and_then(|p| p.output.clone()))
        .or_else(|| defaults.as_ref().map(|d| d.output.clone()))
        .unwrap_or(OutputFormat::Text);

    let mut env = config.map(|cfg| cfg.env.clone()).unwrap_or_default();
    if let Some(profile) = &profile {
        env.extend(profile.env.clone());
    }

    Ok(ResolvedConfig {
        agent_def,
        tools_file,
        skills_dir,
        template,
        max_iterations,
        max_runtime,
        output,
        env,
    })
}

/// The names `atoma.toml` asked to keep out of every tool server.
///
/// A function rather than a field on `ResolvedConfig`, because there is nothing to
/// resolve it against: no CLI flag to lose to and no profile override to merge, and
/// the caller that needs it reaches it before any `ResolvedConfig` would help.
/// `atoma validate --with-live-tools` starts real servers and has no run config at
/// all, so a field on one would have left that path unprotected.
pub fn protected_env_names(config: Option<&AtomaConfig>) -> Vec<String> {
    match config {
        Some(cfg) => cfg.protect_env.clone(),
        None => Vec::new(),
    }
}

/// Generate a default atoma.toml file.
pub fn generate_default_config() -> String {
    r#"# Atoma configuration
# See https://github.com/yuma-seno/atoma for documentation.

# Environment variable names no tool server may inherit, on top of the provider
# keys atoma already strips. Name your own secrets here -- CI tokens, chat tokens,
# anything a `shell` server has no business reading out of its own environment.
#
# A server that legitimately needs one names it in its own `env` in the tools
# file, which is applied after the removal and so puts it back -- for that one
# server and no other.
#
# This key is top-level on purpose. Uncommented below a [table] header it would
# belong to that table and do nothing.
# protect_env = ["CI_JOB_TOKEN", "SLACK_BOT_TOKEN"]

[defaults]
# agent_def = "agents/default.md"
# tools_file = "tools.yaml"
# skills_dir = "skills"
# template = "templates/custom.md"
# output = "text"   # "text" or "json"

# Neither ceiling below is set by default, and a run with neither is unbounded: it
# stops when the agent says it is finished, or when a broken loop is detected.
#
# max_runtime_secs is the one to reach for. A caller running under something that
# will kill it -- a CI job with a timeout -- should set this below that timeout, so
# the run ends on its own terms and saves its session instead of being killed with
# the session unwritten.
# max_runtime_secs = 3000
#
# max_iterations counts turns, which is a proxy for "this run is getting nowhere"
# and a poor one in both directions. Set it only if you want a hard call ceiling.
# max_iterations = 200

# Profile: overrides defaults when --profile is used
# [profile.review]
# agent_def = "agents/reviewer.md"
# output = "json"

# Environment variables to set when running under this config
# [env]
# OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_run_settings_using_cli_profile_defaults_priority() {
        let config: AtomaConfig = toml::from_str(
            r#"
[defaults]
template = "default.md"
skills_dir = "default-skills"

[profile.review]
template = "review.md"
skills_dir = "review-skills"
output = "json"
"#,
        )
        .unwrap();

        let from_profile = resolve_run_config(
            CliOverrides {
                agent_def: None,
                tools_file: None,
                skills_dir: None,
                template: None,
                max_iterations: None,
                max_runtime_secs: None,
                output: None,
            },
            Some("review"),
            Some(&config),
        )
        .unwrap();
        assert_eq!(from_profile.template, Some(PathBuf::from("review.md")));
        assert_eq!(
            from_profile.skills_dir,
            Some(PathBuf::from("review-skills"))
        );
        assert_eq!(from_profile.output, OutputFormat::Json);

        let from_cli = resolve_run_config(
            CliOverrides {
                agent_def: None,
                tools_file: None,
                skills_dir: Some(PathBuf::from("cli-skills")),
                template: Some(PathBuf::from("cli.md")),
                max_iterations: None,
                max_runtime_secs: None,
                output: None,
            },
            Some("review"),
            Some(&config),
        )
        .unwrap();
        assert_eq!(from_cli.template, Some(PathBuf::from("cli.md")));
        assert_eq!(from_cli.skills_dir, Some(PathBuf::from("cli-skills")));
    }

    #[test]
    fn profile_environment_overrides_global_environment() {
        let config: AtomaConfig = toml::from_str(
            r#"
[env]
SHARED = "global"
GLOBAL_ONLY = "yes"

[profile.review.env]
SHARED = "profile"
PROFILE_ONLY = "yes"
"#,
        )
        .unwrap();

        let resolved = resolve_run_config(
            CliOverrides {
                agent_def: None,
                tools_file: None,
                skills_dir: None,
                template: None,
                max_iterations: None,
                max_runtime_secs: None,
                output: None,
            },
            Some("review"),
            Some(&config),
        )
        .unwrap();

        assert_eq!(
            resolved.env.get("SHARED").map(String::as_str),
            Some("profile")
        );
        assert_eq!(
            resolved.env.get("GLOBAL_ONLY").map(String::as_str),
            Some("yes")
        );
        assert_eq!(
            resolved.env.get("PROFILE_ONLY").map(String::as_str),
            Some("yes")
        );
    }

    /// A protected name is a property of the project rather than of the run, so
    /// the key is top-level and there is no profile that can narrow it.
    #[test]
    fn protected_names_are_read_from_the_top_level_key() {
        let raw = r#"protect_env = ["CI_JOB_TOKEN"]"#;
        let config: AtomaConfig = toml::from_str(raw).unwrap();
        assert_eq!(protected_env_names(Some(&config)), vec!["CI_JOB_TOKEN"]);
    }

    /// Every existing atoma.toml predates this key, and none of them names it. An
    /// absent list has to mean "nothing extra", not a parse error -- adding a
    /// protection mechanism that stops every configured project from starting would
    /// be a strange way to make them safer.
    #[test]
    fn a_config_without_the_key_protects_nothing_extra() {
        let config: AtomaConfig = toml::from_str("[defaults]").unwrap();
        assert!(protected_env_names(Some(&config)).is_empty());
        assert!(protected_env_names(None).is_empty());
    }

    #[test]
    fn rejects_unknown_profile() {
        let config: AtomaConfig = toml::from_str("[profile.review]\nmax_iterations = 10").unwrap();
        let result = resolve_run_config(
            CliOverrides {
                agent_def: None,
                tools_file: None,
                skills_dir: None,
                template: None,
                max_iterations: None,
                max_runtime_secs: None,
                output: None,
            },
            Some("typo"),
            Some(&config),
        );

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Profile 'typo' not found"));
    }
}
