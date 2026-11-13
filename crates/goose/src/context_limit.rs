use anyhow::Result;
use oi_kit_providers::context_limit::ContextLimitResolver;

use crate::config::Config;
use crate::providers::base::Provider;

pub async fn get_context_limit(provider: &dyn Provider, model: &str) -> Result<usize> {
    let override_limit = Config::global().get_goose_context_limit()?;
    if override_limit.is_some() {
        return Ok(provider.get_context_limit(model, override_limit).await);
    }
    // Global declarative fallback: if provider has no configured limit for this
    // model (falls to canonical/128k) but another custom_providers entry does,
    // use it. Do not override provider-local configured limits.
    let provider_limit = provider.get_context_limit(model, None).await;
    if provider_limit != goose_providers::model::DEFAULT_CONTEXT_LIMIT {
        return Ok(provider_limit);
    }
    let global_limit = {
        let custom = goose_providers::declarative::load_custom_providers(
            &crate::config::declarative_providers::custom_providers_dir(),
        )
        .unwrap_or_default();
        custom
            .iter()
            .flat_map(|cfg| &cfg.models)
            .find(|m| m.name == model)
            .and_then(|m| m.context_limit)
    };
    Ok(global_limit.unwrap_or(provider_limit))
}

pub fn get_local_context_limit(provider_name: &str, model: &str) -> Result<usize> {
    let override_limit = Config::global().get_goose_context_limit()?;
    let mut configured_limits = Vec::new();

    #[cfg(feature = "aws-providers")]
    if provider_name == "aws_bedrock" {
        if let Some(limit) = crate::providers::bedrock::local_context_limit(model) {
            configured_limits.push((model.to_string(), limit));
        }
    }

    #[cfg(feature = "local-inference")]
    if provider_name == "local" {
        if let Some(limit) = crate::providers::local_inference::local_context_limit(model) {
            configured_limits.push((model.to_string(), limit));
        }
    }

    configured_limits.extend(
        crate::config::declarative_providers::load_provider(provider_name)
            .ok()
            .into_iter()
            .flat_map(|loaded| loaded.config.models)
            .filter_map(|model| model.context_limit.map(|limit| (model.name, limit))),
    );

    // Global declarative fallback: any custom_providers/*.json entry for this
    // model patches the limit regardless of which provider discovered it (EnMaaS
    // single endpoint advertises same IDs on both engines).
    {
        let global_limits = goose_providers::declarative::load_custom_providers(
            &crate::config::declarative_providers::custom_providers_dir(),
        )
        .unwrap_or_default()
        .into_iter()
        .flat_map(|cfg| cfg.models)
        .filter_map(|m| m.context_limit.map(|limit| (m.name, limit)));
        // Prefer provider-local entries: only add global if not already present
        // (exact-case match, mirroring ContextLimitResolver precedence).
        let mut seen: std::collections::HashSet<String> =
            configured_limits.iter().map(|(n, _)| n.clone()).collect();
        for (name, limit) in global_limits {
            if seen.insert(name.clone()) {
                configured_limits.push((name, limit));
            }
        }
    }

    Ok(ContextLimitResolver::new(provider_name)
        .with_configured_limits(configured_limits)
        .resolve_local(model, override_limit))
}
