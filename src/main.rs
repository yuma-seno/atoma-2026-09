mod application;
mod cli;
mod domain;
mod infra;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use crate::application::runner::{CompletionReason, RunDeps, RunOutcome, RunSettings};
use crate::cli::{Cli, Command};
use crate::domain::ports::AgentDefPort;
use crate::infra::config::{self as config_module, CliOverrides, OutputFormat};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    // Before anything reads a credential, and before any tool server exists to
    // read this process. See infra/process_protection.rs for what it closes and
    // what it deliberately does not.
    if cli.no_process_protection {
        tracing::warn!(
            "process protection disabled: a tool server running as this user can read this process's environment and memory, including the provider API key"
        );
    } else {
        crate::infra::process_protection::harden_against_same_user_inspection();
    }

    // Before the command runs, so the file is gone before anything could read it,
    // and so a malformed one fails before any work is done.
    let credentials = match cli.credentials_file.as_deref() {
        Some(path) => infra::credentials::Credentials::from_file(path)?,
        None => infra::credentials::Credentials::from_environment(),
    };

    match cli.command {
        Command::Run {
            agent_def,
            profile,
            output,
            in_session,
            prompt_file,
            out_session,
            template,
            tools_file,
            skills_dir,
            max_iterations,
            max_runtime_secs,
            stop_file,
            fail_on_tool_findings,
        } => {
            let config = config_module::discover_and_load()?.1;

            let output_override = output.as_deref().map(OutputFormat::from_arg).transpose()?;

            let resolved = config_module::resolve_run_config(
                CliOverrides {
                    agent_def,
                    tools_file,
                    skills_dir,
                    template,
                    max_iterations,
                    max_runtime_secs,
                    output: output_override,
                },
                profile.as_deref(),
                config.as_ref(),
            )?;
            let output_format = resolved.output.clone();

            // Before the loop below, which may put one of these names into this
            // process's own environment, and long before anything exists that could
            // start a tool server. `declare_protected_env_names` says why the list
            // is registered rather than carried to the one place that spawns.
            let protect = config_module::protected_env_names(config.as_ref());
            infra::credentials::declare_protected_env_names(protect);

            for (key, value) in &resolved.env {
                if std::env::var(key).is_err() {
                    std::env::set_var(key, value);
                }
            }

            let agent_def_port = infra::persistence::agent_def::FileAgentDefAdapter;
            let parsed_agent = AgentDefPort::parse(&agent_def_port, &resolved.agent_def)?;
            let provider_hint = parsed_agent.frontmatter.provider.as_deref();
            let llm = infra::llm::build_llm_client(
                provider_hint,
                &credentials,
                &parsed_agent.frontmatter.extra_headers,
            )
            .await?;
            let session_port = infra::persistence::session::FileSessionAdapter;
            // Moved in after the client is built: the adapter owns the credentials
            // from here on, and is what routes each one to the single tool server
            // whose `env` names it.
            let tool_def_port = infra::persistence::tool_def::FileToolDefAdapter::new(credentials);
            let skill_port = infra::persistence::skill::FileSkillAdapter;
            // Straight from the flag, like `stop_file`: what a run does about a
            // configuration finding belongs to the invocation, not to the agent. See
            // `Command::Run::fail_on_tool_findings` for why it is off by default.
            let mcp_factory = infra::mcp::McpRegistryFactory::new(fail_on_tool_findings);

            let result = application::runner::run(
                RunSettings {
                    agent_def_path: resolved.agent_def,
                    in_session,
                    prompt_file,
                    out_session,
                    template_path: resolved.template,
                    tools_file: resolved.tools_file,
                    skills_dir: resolved.skills_dir,
                    max_iterations: resolved.max_iterations,
                    max_runtime: resolved.max_runtime,
                    // Straight from the flag, not through `resolve_run_config`: there
                    // is no config key for it, and there should not be. See
                    // `RunSettings::stop_file`.
                    stop_file,
                },
                RunDeps {
                    llm: llm.as_ref(),
                    template: &infra::template::FileTemplateAdapter,
                    agent_def: &agent_def_port,
                    session: &session_port,
                    tool_def: &tool_def_port,
                    skill: &skill_port,
                    mcp_factory: &mcp_factory,
                },
            )
            .await;

            let outcome = match result {
                Ok(outcome) => outcome,
                Err(err) if application::runner::is_soft_stop(&err) => {
                    std::process::exit(2);
                }
                Err(err) => return Err(err),
            };

            if let RunOutcome::Completed {
                text,
                usage,
                reason,
                session_path,
            } = outcome
            {
                match output_format {
                    OutputFormat::Text => println!("{}", text),
                    OutputFormat::Json => {
                        let output = serde_json::json!({
                            "response": text,
                            "usage": {
                                "prompt_tokens": usage.prompt_tokens,
                                "completion_tokens": usage.completion_tokens,
                                "total_tokens": usage.total_tokens,
                            },
                            "finish_reason": match reason {
                                CompletionReason::Stop => "stop",
                                CompletionReason::Length => "length",
                            },
                            "session_path": session_path.map(|p| p.to_string_lossy().to_string()),
                        });
                        println!("{}", serde_json::to_string(&output)?);
                    }
                }
            }

            Ok(())
        }
        Command::Validate {
            agent_def,
            tools_file,
            template,
            credentials_present,
            with_live_tools,
        } => {
            let agent_def_port = infra::persistence::agent_def::FileAgentDefAdapter;
            let tool_def_port = infra::persistence::tool_def::FileToolDefAdapter::default();
            application::validator::validate(
                agent_def.clone(),
                tools_file.clone(),
                template,
                &agent_def_port,
                &tool_def_port,
            )?;
            // A separate question, asked only when a caller opts in. See
            // `validate_credentials` for why it is not a branch inside `validate`.
            if let Some(raw) = credentials_present {
                application::validator::validate_credentials(
                    &agent_def,
                    &application::validator::credentials_from_arg(&raw),
                    &agent_def_port,
                )?;
            }
            // Also opt-in, and for the same reason: it needs more than the default
            // check has. See `validate_live_tools`.
            if with_live_tools {
                // This starts real tool servers with this process's environment, so
                // the caller's declared names have to reach it too -- it is the
                // second spawning path, and the one a change to the run path would
                // silently leave behind.
                //
                // Read here rather than at the top of the arm: a plain `atoma
                // validate` starts nothing, and should not begin failing because an
                // unrelated key in atoma.toml will not parse.
                let config = config_module::discover_and_load()?.1;
                let protect = config_module::protected_env_names(config.as_ref());
                infra::credentials::declare_protected_env_names(protect);

                match tools_file {
                    Some(path) => {
                        application::validator::validate_live_tools(
                            &agent_def,
                            &path,
                            &agent_def_port,
                            &tool_def_port,
                        )
                        .await?
                    }
                    None => anyhow::bail!(
                        "--with-live-tools needs --tools-file: the servers it starts are \
                         the ones that file declares."
                    ),
                }
            }
            Ok(())
        }
        Command::Init => {
            let template = config_module::generate_default_config();
            println!("{}", template);
            Ok(())
        }
    }
}
