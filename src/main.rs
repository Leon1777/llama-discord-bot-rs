mod bot;
mod commands;

use bot::default_prompt::SYSTEM_PROMPT;
use bot::inference::Message as LlmMessage;
use commands::{ask::ask, mission::mission, reset::reset};
use dotenv::dotenv;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel},
};
use poise::serenity_prelude as serenity;
use poise::serenity_prelude::ActivityData;
use poise::Framework;
use serenity::model::gateway::GatewayIntents;
use std::{
    env,
    process::Command,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{sync::Mutex as TokioMutex, task, time::sleep};

#[derive(Debug)]
pub struct BotContext {
    pub chat_history: Arc<Mutex<Vec<LlmMessage>>>,
    pub request_lock: Arc<TokioMutex<()>>,
    pub model: Arc<LlamaModel>,
    pub backend: Arc<LlamaBackend>,
}

#[derive(Debug)]
struct BotError;

impl std::fmt::Display for BotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "An error occurred in the bot")
    }
}

impl std::error::Error for BotError {}

impl From<serenity::Error> for BotError {
    fn from(_: serenity::Error) -> Self {
        BotError
    }
}

// GPU stats using nvidia-smi
async fn get_gpu_stats() -> String {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu")
        .arg("--format=csv,noheader,nounits")
        .output();

    match output {
        Ok(output) => {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = output_str.trim().split(", ").collect();

            if parts.len() == 5 {
                return format!(
                    "GPU: {} | {}% | {} MB / {} MB | {}°C",
                    parts[0], parts[1], parts[2], parts[3], parts[4]
                );
            }
            "Failed to parse GPU stats".to_string()
        }
        Err(_) => "Failed to retrieve GPU info".to_string(),
    }
}

async fn monitor_gpu_stats(ctx: serenity::Context) {
    loop {
        let gpu_status = get_gpu_stats().await;
        ctx.set_activity(Some(ActivityData::listening(&gpu_status)));
        sleep(Duration::from_secs(10)).await;
    }
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    let token = env::var("DISCORD_TOKEN").expect("Expected DISCORD_TOKEN in environment variables");
    let intents = GatewayIntents::GUILD_MESSAGES | GatewayIntents::MESSAGE_CONTENT;

    // Initialize backend and model
    let backend = Arc::new(LlamaBackend::init().expect("Failed to initialize Llama backend"));

    let model_path = "models/Mistral-Nemo-Instruct-2407-abliterated_q8_0.gguf";
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(41)
        .with_use_mlock(true);

    let model = Arc::new(
        LlamaModel::load_from_file(&backend, model_path, &model_params)
            .expect("Failed to load Llama model"),
    );

    let bot_state = Arc::new(BotContext {
        chat_history: Arc::new(Mutex::new(vec![LlmMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        }])),
        request_lock: Arc::new(TokioMutex::new(())),
        model,
        backend,
    });

    let framework = Framework::builder()
        .setup(|ctx, ready, _framework| {
            Box::pin(async move {
                println!("Logged in as {}", ready.user.name);

                // GPU monitoring in background
                let ctx_clone = ctx.clone();
                task::spawn(async move {
                    monitor_gpu_stats(ctx_clone).await;
                });

                Ok(bot_state)
            })
        })
        .options(poise::FrameworkOptions::<Arc<BotContext>, BotError> {
            commands: vec![ask(), reset(), mission()],
            on_error: |error| {
                Box::pin(async move {
                    println!("Error: {:?}", error);
                })
            },
            ..Default::default()
        })
        .build();

    let mut client = serenity::ClientBuilder::new(token, intents)
        .framework(framework)
        .await
        .expect("Error creating client");

    if let Err(why) = client.start().await {
        println!("Client error: {:?}", why);
    }
}
