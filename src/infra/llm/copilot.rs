use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::domain::ports::{LlmPort, LlmResponse};
use crate::domain::session::Message;
use crate::infra::llm::shared::{chat_response_to_llm, openai_compat_call};
/// Exchange a GitHub PAT for a short-lived GitHub Copilot API token.
async fn exchange_copilot_token(client: &reqwest::Client, github_token: &str) -> Result<String> {
    const COPILOT_AUTH_URL: &str = "https://api.github.com/copilot_internal/v2/token";

    tracing::debug!("Exchanging GitHub token for Copilot token");

    #[derive(Deserialize)]
    struct TokenResponse {
        token: String,
    }

    let response = client
        .get(COPILOT_AUTH_URL)
        .header("Authorization", format!("token {}", github_token))
        .header("Accept", "application/json")
        .send()
        .await
        .context("Failed to request GitHub Copilot token")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "unknown error".to_string());
        anyhow::bail!(
            "Failed to obtain GitHub Copilot token ({}): {}\n\
             Ensure your GitHub token has the 'copilot' scope and a Copilot subscription is active.",
            status,
            error_text
        );
    }

    let resp: TokenResponse = response
        .json()
        .await
        .context("Failed to parse GitHub Copilot token response")?;

    tracing::debug!("Successfully obtained Copilot token");
    Ok(resp.token)
}

/// Client for GitHub Copilot (OpenAI-compatible wire protocol with Copilot auth).
pub struct CopilotClient {
    pub(crate) client: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) copilot_token: String,
}

impl CopilotClient {
    /// Exchange a GitHub credential for a Copilot token, then hold both it and the
    /// endpoint.
    ///
    /// The endpoint is a parameter now. It was a `const` — the only provider whose
    /// address could not be changed, for no reason anyone had written down.
    ///
    /// The fallback to `GITHUB_TOKEN`/`GH_TOKEN` stays, and deliberately does not
    /// take part in provider detection: a run that talks to GitHub has one of those
    /// anyway, so detecting Copilot from them would make every run ambiguous. They
    /// work when this provider was asked for by name.
    pub async fn connect(
        client: reqwest::Client,
        base_url: String,
        headers: Vec<(String, String)>,
        github_token: String,
    ) -> Result<Self> {
        let copilot_token = exchange_copilot_token(&client, &github_token).await?;
        Ok(CopilotClient {
            client,
            base_url,
            headers,
            copilot_token,
        })
    }
}

#[async_trait]
impl LlmPort for CopilotClient {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
        extra_body: &std::collections::HashMap<String, Value>,
    ) -> Result<LlmResponse> {
        let reply = openai_compat_call(
            &self.client,
            &self.base_url,
            &self.copilot_token,
            &self.headers,
            model,
            messages,
            tools,
            extra_body,
        )
        .await?;

        Ok(chat_response_to_llm(reply.body, reply.request_id))
    }
}
