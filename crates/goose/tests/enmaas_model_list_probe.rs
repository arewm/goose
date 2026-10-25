//! Manual probe: fetch the model list from the real EnMaaS endpoints using the
//! same declarative provider code paths goose uses. Run with:
//! ENMAAS_KEY=... cargo test -p goose --test enmaas_model_list_probe -- --nocapture --ignored

use goose::config::Config;
use goose::providers::base::Provider;
use goose::providers::custom_provider_config::ConfigKeyResolver;
use goose_providers::declarative::{load_custom_providers, ProviderEngine};

#[tokio::test]
#[ignore]
async fn fetch_models_from_enmaas_providers() {
    let dir = dirs::home_dir()
        .expect("home dir")
        .join(".config/goose/custom_providers");
    let providers = load_custom_providers(&dir).expect("load custom providers");
    let plain = reqwest::get("https://api.enmaas.devshift.net/v1/models").await;
    println!("[plain reqwest] {:?}", plain.map(|r| r.status()));
    for config in providers {
        let name = config.name.clone();
        let engine = format!("{:?}", config.engine);
        let result = match config.engine {
            ProviderEngine::OpenAI => {
                goose::providers::openai_def::from_custom_config(config.clone(), None)
                    .map(|p| p.fetch_boxed())
            }
            ProviderEngine::Anthropic => goose_providers::anthropic::from_declarative_config(
                config.clone(),
                None,
                ConfigKeyResolver::new(Config::global()),
            )
            .map(|b| b.build().fetch_boxed()),
            _ => continue,
        };
        match result {
            Ok(provider) => match provider.fetch_supported_models().await {
                Ok(models) => println!("[{name} ({engine})] OK ({}): {:?}", models.len(), models),
                Err(e) => println!("[{name} ({engine})] ERR: {e}"),
            },
            Err(e) => println!("[{name} ({engine})] construct failed: {e}"),
        }
    }
}

trait FetchBoxed {
    fn fetch_boxed(self) -> Box<dyn Provider>;
}

impl FetchBoxed for goose_providers::openai::OpenAiProvider {
    fn fetch_boxed(self) -> Box<dyn Provider> {
        Box::new(self)
    }
}

impl FetchBoxed for goose_providers::anthropic::AnthropicProvider {
    fn fetch_boxed(self) -> Box<dyn Provider> {
        Box::new(self)
    }
}
