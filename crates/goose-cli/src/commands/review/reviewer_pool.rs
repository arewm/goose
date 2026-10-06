use anyhow::{bail, Context, Result};
use goose::checks::{Check, DiscoveredReview};
use goose::config::{Config, ConfigError};
use rand::RngExt;
use serde::Deserialize;
use std::collections::HashSet;

use super::handler::ReviewOptions;

pub const REVIEWER_POOL_KEY: &str = "GOOSE_REVIEWER_POOL";

const PROVIDERS_WITH_NATIVE_TOOLS: &[&str] = &[
    "amp-acp",
    "chatgpt-codex",
    "claude-acp",
    "claude-code",
    "codex",
    "codex-acp",
    "copilot-acp",
    "cursor-agent",
    "gemini-cli",
    "pi-acp",
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reviewer {
    pub name: String,
    pub provider: Option<String>,
    pub model: Option<String>,
}

impl Reviewer {
    pub fn as_check(
        &self,
        review_prompt: &str,
        discovered: &DiscoveredReview,
        user_text: Option<&str>,
    ) -> Check {
        Check {
            name: self.name.clone(),
            description: None,
            model: self.model.clone(),
            turn_limit: None,
            tools: Some(Vec::new()),
            severity_default: None,
            path: Default::default(),
            scope_dir: String::new(),
            body: pool_review_prompt(review_prompt, discovered, user_text),
        }
    }
}

fn pool_review_prompt(
    base_prompt: &str,
    discovered: &DiscoveredReview,
    user_text: Option<&str>,
) -> String {
    let base_prompt = if base_prompt == super::prompt::DEFAULT_REVIEW_PROMPT {
        strip_default_prompt_sections(base_prompt)
    } else {
        base_prompt.trim_end().to_string()
    };
    let mut prompt = String::new();
    if !base_prompt.trim().is_empty() {
        prompt.push_str(base_prompt.trim_end());
        prompt.push_str("\n\n");
    }
    if let Some(text) = user_text.map(str::trim).filter(|text| !text.is_empty()) {
        prompt.push_str("## Additional user review context\n\n");
        prompt.push_str(text);
        prompt.push_str("\n\n");
    }
    if !discovered.checks.is_empty() {
        prompt.push_str("## Applicable repository review instructions\n\n");
        for check in &discovered.checks {
            let scope = if check.scope_dir.is_empty() {
                "<root>"
            } else {
                &check.scope_dir
            };
            prompt.push_str(&format!(
                "### {} (scope: {})\n\n{}\n\n",
                check.name,
                scope,
                check.body.trim()
            ));
        }
    }
    prompt.push_str("Review the entire diff as one reviewer. Do not delegate, synthesize other reviews, or use tools. Return ONLY valid JSON with the exact schema {\"findings\":[{\"severity\":\"low|medium|high|critical\",\"path\":\"relative/path\",\"line_start\":1,\"line_end\":1,\"summary\":\"Actionable issue and fix\"}]}. If no findings, return {\"findings\":[]}. Do not include any text outside the JSON object.");
    prompt
}

fn strip_default_prompt_sections(prompt: &str) -> String {
    let mut result = String::new();
    let mut skip = false;
    for line in prompt.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            skip = matches!(heading, "Output" | "Checks");
        }
        if !skip {
            result.push_str(line);
            result.push('\n');
        }
    }
    let result = result
        .replace(
            "Before delegating to subagent checks, do a careful correctness pass on the\ndiff yourself. Walk every changed function and look hard for:",
            "Walk every changed function and look hard for:",
        )
        .replace(
            "Emit findings from this pass with `\"check\": \"main\"`.",
            "",
        );
    result.trim_end().to_string()
}

pub fn load_pool(config: &Config) -> Result<Vec<Reviewer>> {
    let value = match config.get_param::<serde_json::Value>(REVIEWER_POOL_KEY) {
        Ok(value) => value,
        Err(ConfigError::NotFound(_)) => return Ok(Vec::new()),
        Err(error) => return Err(error).with_context(|| format!("invalid {REVIEWER_POOL_KEY}")),
    };
    let pool: Vec<Reviewer> = serde_json::from_value(value).with_context(|| {
        format!("invalid {REVIEWER_POOL_KEY}: expected a list of reviewer entries")
    })?;
    validate_pool(&pool)?;
    Ok(pool)
}

fn validate_pool(pool: &[Reviewer]) -> Result<()> {
    let mut names = HashSet::new();
    for reviewer in pool {
        if reviewer.name.trim().is_empty() {
            bail!("{REVIEWER_POOL_KEY}: each reviewer needs a nonempty name");
        }
        if !names.insert(reviewer.name.trim()) {
            bail!(
                "{REVIEWER_POOL_KEY}: duplicate reviewer name '{}'",
                reviewer.name
            );
        }
        for (key, value) in [("provider", &reviewer.provider), ("model", &reviewer.model)] {
            if value.as_ref().is_some_and(|value| value.trim().is_empty()) {
                bail!(
                    "{REVIEWER_POOL_KEY}: reviewer '{}' has an empty {key}",
                    reviewer.name
                );
            }
        }
    }
    Ok(())
}

/// The selector is called once with the pool size and must return an index in that range.
pub fn select_reviewer(
    pool: &[Reviewer],
    select_index: impl FnOnce(usize) -> usize,
) -> Option<&Reviewer> {
    if pool.is_empty() {
        return None;
    }
    Some(&pool[select_index(pool.len())])
}

pub fn random_reviewer(pool: &[Reviewer]) -> Option<&Reviewer> {
    select_reviewer(pool, |len| rand::rng().random_range(0..len))
}

fn reviewer_overrides(
    reviewer: &Reviewer,
    opts: &ReviewOptions,
    configured_provider: Option<&str>,
) -> (Option<String>, Option<String>) {
    let provider = reviewer
        .provider
        .clone()
        .or_else(|| opts.provider.clone())
        .or_else(|| configured_provider.map(str::to_owned));
    let same_provider = reviewer.provider.is_none()
        || reviewer.provider == opts.provider
        || (opts.provider.is_none() && reviewer.provider.as_deref() == configured_provider);
    let model = opts
        .override_model
        .clone()
        .or_else(|| reviewer.model.clone())
        .or_else(|| same_provider.then(|| opts.default_model.clone()).flatten());
    (provider, model)
}

pub async fn resolve_provider_model(
    reviewer: &Reviewer,
    opts: &ReviewOptions,
    config: &Config,
) -> Result<(String, String)> {
    let configured_provider = config.get_goose_provider().ok();
    let (provider, model) = reviewer_overrides(reviewer, opts, configured_provider.as_deref());
    let provider =
        provider.context("No review provider configured. Run 'goose configure' first.")?;
    if PROVIDERS_WITH_NATIVE_TOOLS.contains(&provider.as_str()) {
        bail!(
            "Reviewer-pool mode rejects provider '{provider}' because it runs an external coding agent with native tools outside Goose's chat-only and MCP controls."
        );
    }
    let model = model
        .or_else(|| {
            (configured_provider.as_deref() == Some(provider.as_str()))
                .then(|| config.get_goose_model().ok())
                .flatten()
        })
        .or_else(|| {
            goose::config::get_provider_entry(config, &provider)
                .map(|entry| entry.model)
                .filter(|model| !model.is_empty())
        });
    let model = match model {
        Some(model) => model,
        None => goose::providers::get_from_registry(&provider)
            .await?
            .metadata()
            .default_model
            .clone(),
    };
    if model.trim().is_empty() {
        bail!("No review model configured for provider '{provider}'");
    }
    Ok((provider, model))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reviewer(name: &str) -> Reviewer {
        Reviewer {
            name: name.into(),
            provider: None,
            model: None,
        }
    }

    #[test]
    fn selection_uses_exactly_one_pool_entry() {
        let pool = vec![reviewer("first"), reviewer("second"), reviewer("third")];
        for index in 0..pool.len() {
            let mut calls = 0;
            let selected = select_reviewer(&pool, |len| {
                calls += 1;
                assert_eq!(len, 3);
                index
            })
            .unwrap();
            assert_eq!(selected.name, pool[index].name);
            assert_eq!(calls, 1);
        }
        assert!(select_reviewer(&[], |_| panic!("empty pool must not select")).is_none());
    }

    #[test]
    fn reviewer_prompt_contains_diff_override_and_repository_rules() {
        use goose::checks::Check;
        use std::path::PathBuf;

        let pool = vec![reviewer("first"), reviewer("second")];
        let selected = select_reviewer(&pool, |_| 1).unwrap();
        let discovered = DiscoveredReview {
            checks: vec![Check {
                name: "repo-rules".into(),
                description: None,
                model: None,
                turn_limit: None,
                tools: None,
                severity_default: None,
                path: PathBuf::from(".agents/REVIEW.md"),
                scope_dir: "".into(),
                body: "Follow repository rules".into(),
            }],
        };
        let default_prompt = include_str!("default_review_prompt.md");
        let adapted_default_prompt = pool_review_prompt(default_prompt, &discovered, None);
        assert!(adapted_default_prompt.contains("correctness bugs"));
        assert!(!adapted_default_prompt.contains("single line containing `[]`"));
        assert!(!adapted_default_prompt.contains("Dispatch them all in parallel"));
        assert!(adapted_default_prompt.contains("Follow repository rules"));
        assert!(adapted_default_prompt.contains("Return ONLY valid JSON"));
        assert!(adapted_default_prompt.contains("\"findings\":[]"));
        let check = selected.as_check(
            "User-supplied review prompt",
            &discovered,
            Some("Focus on authorization regressions"),
        );
        assert_eq!(check.name, "second");
        assert_eq!(check.tools, Some(Vec::new()));
        assert!(check.body.contains("User-supplied review prompt"));
        assert!(check.body.contains("Focus on authorization regressions"));
        assert!(check.body.contains("Follow repository rules"));
        assert!(check
            .body
            .contains("Do not include any text outside the JSON object"));
        let diff = "diff --git a/one b/one\n+one\ndiff --git a/two b/two\n+two";
        let prompt = super::super::orchestrator::build_check_prompt(&check, diff, None, 7, true);
        assert!(prompt.contains(diff));
        assert!(prompt.contains("User-supplied review prompt"));
        assert!(prompt.contains("Focus on authorization regressions"));
        assert!(prompt.contains("Follow repository rules"));
        assert!(prompt.contains("Do not delegate"));
    }

    #[test]
    fn validates_all_entries_before_selection() {
        assert!(validate_pool(&[]).is_ok());
        assert!(validate_pool(&[reviewer("valid")]).is_ok());
        assert!(validate_pool(&[reviewer("same"), reviewer("same")]).is_err());
        for field in ["name", "provider", "model"] {
            let mut entry = reviewer("valid");
            match field {
                "name" => entry.name = " ".into(),
                "provider" => entry.provider = Some(" ".into()),
                "model" => entry.model = Some(String::new()),
                _ => unreachable!(),
            }
            assert!(
                validate_pool(&[reviewer("first"), entry]).is_err(),
                "{field}"
            );
        }
    }

    #[test]
    fn configured_values_win_over_session_fallbacks() {
        let opts = ReviewOptions {
            provider: Some("session-provider".into()),
            default_model: Some("session-model".into()),
            ..Default::default()
        };
        let mut entry = reviewer("configured");
        assert_eq!(
            reviewer_overrides(&entry, &opts, None),
            (opts.provider.clone(), opts.default_model.clone())
        );
        entry.model = Some("review-model".into());
        assert_eq!(
            reviewer_overrides(&entry, &opts, None).1.as_deref(),
            Some("review-model")
        );
        entry.provider = Some("review-provider".into());
        assert_eq!(
            reviewer_overrides(&entry, &opts, None),
            (entry.provider.clone(), entry.model.clone())
        );
        entry.model = None;
        assert_eq!(reviewer_overrides(&entry, &opts, None).1, None);
        entry.provider = opts.provider.clone();
        assert_eq!(
            reviewer_overrides(&entry, &opts, None).1,
            opts.default_model
        );
        let opts = ReviewOptions {
            override_model: Some("forced-model".into()),
            ..opts
        };
        entry.model = Some("review-model".into());
        assert_eq!(
            reviewer_overrides(&entry, &opts, None).1,
            opts.override_model
        );

        let standalone_opts = ReviewOptions {
            provider: None,
            default_model: Some("cli-model".into()),
            ..Default::default()
        };
        let mut configured_same_provider = reviewer("configured");
        configured_same_provider.provider = Some("openai".into());
        assert_eq!(
            reviewer_overrides(&configured_same_provider, &standalone_opts, Some("openai")),
            (Some("openai".into()), Some("cli-model".into()))
        );
    }

    #[tokio::test]
    async fn resolves_standalone_defaults_and_provider_specific_models() {
        let _env = env_lock::lock_env([
            ("GOOSE_PROVIDER", None::<&str>),
            ("GOOSE_MODEL", None::<&str>),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let config = Config::new_with_file_secrets(
            dir.path().join("config.yaml"),
            dir.path().join("secrets.yaml"),
        )
        .unwrap();
        config.set_goose_provider("openai").unwrap();
        config.set_goose_model("configured-model").unwrap();
        let mut entry = reviewer("default");
        assert_eq!(
            resolve_provider_model(&entry, &ReviewOptions::default(), &config)
                .await
                .unwrap(),
            ("openai".into(), "configured-model".into())
        );
        goose::config::set_provider_entry(
            &config,
            "anthropic",
            &goose::config::ProviderEntry {
                enabled: true,
                configured: true,
                model: "anthropic-configured-model".into(),
            },
        )
        .unwrap();
        entry.provider = Some("anthropic".into());
        let opts = ReviewOptions {
            provider: Some("openai".into()),
            default_model: Some("interactive-model".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_provider_model(&entry, &opts, &config)
                .await
                .unwrap(),
            ("anthropic".into(), "anthropic-configured-model".into())
        );
        entry.provider = Some("openai".into());
        assert_eq!(
            resolve_provider_model(
                &entry,
                &ReviewOptions {
                    provider: None,
                    default_model: Some("command-model".into()),
                    ..Default::default()
                },
                &config
            )
            .await
            .unwrap(),
            ("openai".into(), "command-model".into())
        );
        entry.provider = Some("anthropic".into());
        entry.model = Some("reviewer-model".into());
        assert_eq!(
            resolve_provider_model(&entry, &opts, &config)
                .await
                .unwrap(),
            ("anthropic".into(), "reviewer-model".into())
        );

        entry.provider = Some("codex".into());
        assert!(resolve_provider_model(&entry, &opts, &config)
            .await
            .unwrap_err()
            .to_string()
            .contains("native tools"));
    }

    #[test]
    fn missing_and_empty_config_fall_back_but_malformed_config_errors() {
        let _env = env_lock::lock_env([(REVIEWER_POOL_KEY, None::<&str>)]);
        let dir = tempfile::tempdir().unwrap();
        let config = Config::new_with_file_secrets(
            dir.path().join("config.yaml"),
            dir.path().join("secrets.yaml"),
        )
        .unwrap();
        assert!(load_pool(&config).unwrap().is_empty());
        for yaml in ["[]", "- name: safe\n"] {
            config
                .set_param(
                    REVIEWER_POOL_KEY,
                    serde_yaml::from_str::<serde_yaml::Value>(yaml).unwrap(),
                )
                .unwrap();
            assert!(load_pool(&config).is_ok());
        }
        for yaml in [
            "null",
            "{}",
            "not-a-list",
            "- name: unsafe\n  tools: [summon]",
        ] {
            config
                .set_param(
                    REVIEWER_POOL_KEY,
                    serde_yaml::from_str::<serde_yaml::Value>(yaml).unwrap(),
                )
                .unwrap();
            assert!(load_pool(&config).is_err(), "{yaml}");
        }
    }
}
