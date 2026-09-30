//! Shared model-name resolution against provider catalogs.
//!
//! Used by both the CLI `/model` command and the summon (subagent) extension
//! so that a requested model string is never accepted verbatim: it is either
//! resolved to a concrete catalog entry, reported as ambiguous with the
//! matching candidates, or rejected with suggestions. This prevents switching
//! a session (or delegating a subagent) to a model ID that no provider
//! serves, which previously surfaced only as an opaque request-time 404.

/// Lowercase alphanumeric-only form of a model name, so "GPT-4_Opus" and
/// "gpt4opus" compare equal.
pub fn normalize_model_name(model: &str) -> String {
    model
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

/// Split a model name into lowercase word and number tokens, merging adjacent
/// numeric parts so "5.6", "5-6", and "56" all yield "56".
pub fn model_tokens(model: &str) -> Vec<String> {
    let lower = model.to_lowercase();
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_is_digits = false;
    for c in lower.chars() {
        if c.is_ascii_alphanumeric() {
            let is_digit = c.is_ascii_digit();
            if !current.is_empty() && current_is_digits != is_digit {
                parts.push(std::mem::take(&mut current));
            }
            current_is_digits = is_digit;
            current.push(c);
        } else if !current.is_empty() {
            parts.push(std::mem::take(&mut current));
            current_is_digits = false;
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    let mut tokens: Vec<String> = Vec::new();
    for part in parts {
        if part.chars().all(|c| c.is_ascii_digit()) {
            if let Some(last) = tokens.last_mut() {
                if last.chars().all(|c| c.is_ascii_digit()) {
                    last.push_str(&part);
                    continue;
                }
            }
        }
        tokens.push(part);
    }
    tokens
}

/// Token-level match: every requested token appears in the model's tokens
/// (numeric tokens may be a prefix, so "5" matches "56"). This matches
/// "luna 5.6" against "gpt-5.6-luna" regardless of word order.
pub fn tokens_match(requested: &[String], model: &str) -> bool {
    let model_tokens = model_tokens(model);
    requested.iter().all(|token| {
        model_tokens.iter().any(|candidate| {
            candidate == token
                || (token.chars().all(|c| c.is_ascii_digit())
                    && candidate.starts_with(token.as_str()))
        })
    })
}

fn catalog_matches(requested: &str, catalog: &[String]) -> Vec<String> {
    let normalized = normalize_model_name(requested);
    if normalized.is_empty() {
        return Vec::new();
    }
    if let Some(exact) = catalog
        .iter()
        .find(|model| normalize_model_name(model) == normalized)
    {
        return vec![exact.clone()];
    }
    let requested_tokens = model_tokens(requested);
    let mut candidates: Vec<String> = catalog
        .iter()
        .filter(|model| {
            let model_normalized = normalize_model_name(model);
            (model_normalized.len() > normalized.len() && model_normalized.contains(&normalized))
                || tokens_match(&requested_tokens, model)
        })
        .cloned()
        .collect();
    candidates.sort();
    candidates.dedup();
    candidates
}

/// All catalog entries matching `requested`, best (most advanced variant)
/// last. An exact normalized match collapses to that single entry.
pub fn matching_models(requested: &str, catalog: &[String]) -> Vec<String> {
    catalog_matches(requested, catalog)
}

/// Resolve a requested model name against a provider's catalog: an exact
/// (normalized) match wins; otherwise the lexicographically greatest model
/// whose name contains the request is chosen, so a family name like "opus"
/// or "luna" resolves to its most advanced variant.
pub fn best_catalog_match(requested: &str, catalog: &[String]) -> Option<String> {
    catalog_matches(requested, catalog).pop()
}

/// Closest catalog entries by shared token, for error messages when nothing
/// matched. Each suggestion is formatted as "model (provider)".
pub fn token_shared_suggestions(
    requested: &str,
    catalogs: &[(String, Vec<String>)],
) -> Vec<String> {
    let requested_tokens = model_tokens(requested);
    let mut suggestions: Vec<String> = Vec::new();
    for (name, catalog) in catalogs {
        for model in catalog {
            let shares_token = model_tokens(model).iter().any(|token| {
                requested_tokens.iter().any(|requested| {
                    token == requested
                        || (token.chars().all(|c| c.is_ascii_digit())
                            && token.starts_with(requested.as_str()))
                })
            });
            if shares_token {
                suggestions.push(format!("{model} ({name})"));
            }
        }
    }
    suggestions.sort();
    suggestions.dedup();
    suggestions.truncate(8);
    suggestions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_catalog_match_prefers_exact_then_most_advanced_variant() {
        let catalog = vec![
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "claude-opus-4".to_string(),
            "claude-sonnet-4".to_string(),
        ];
        assert_eq!(
            best_catalog_match("gpt-4o", &catalog),
            Some("gpt-4o".to_string())
        );
        assert_eq!(
            best_catalog_match("opus", &catalog),
            Some("claude-opus-4".to_string())
        );
        assert_eq!(
            best_catalog_match("CLAUDE_OPUS_4", &catalog).as_deref(),
            Some("claude-opus-4")
        );
    }

    #[test]
    fn best_catalog_match_handles_version_qualified_family_names() {
        let catalog = vec![
            "gpt-5.6-luna".to_string(),
            "gpt-6-luna".to_string(),
            "gpt-5.6-sol".to_string(),
        ];
        assert_eq!(
            best_catalog_match("luna 5.6", &catalog),
            Some("gpt-5.6-luna".to_string())
        );
        assert_eq!(
            best_catalog_match("luna", &catalog),
            Some("gpt-6-luna".to_string())
        );
    }

    #[test]
    fn matching_models_returns_all_variants_best_last() {
        let catalog = vec![
            "gpt-5.6-sol".to_string(),
            "gpt-6.1-sol".to_string(),
            "gpt-6-sol".to_string(),
            "glm-5-3".to_string(),
        ];
        assert_eq!(
            matching_models("sol", &catalog),
            vec!["gpt-5.6-sol", "gpt-6-sol", "gpt-6.1-sol"]
        );
        assert_eq!(matching_models("glm", &catalog), vec!["glm-5-3"]);
        assert!(matching_models("opus", &catalog).is_empty());
    }

    #[test]
    fn token_shared_suggestions_share_a_token() {
        let catalogs = vec![(
            "enmaas-openai".to_string(),
            vec!["gpt-6.1-sol".to_string(), "glm-5-3".to_string()],
        )];
        assert_eq!(
            token_shared_suggestions("sol", &catalogs),
            vec!["gpt-6.1-sol (enmaas-openai)".to_string()]
        );
        assert!(token_shared_suggestions("opus", &catalogs).is_empty());
    }
}
