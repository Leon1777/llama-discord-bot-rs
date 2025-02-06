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
    pub system_prompt: Arc<Mutex<String>>,
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
        .arg("--query-gpu=name,temperature.gpu,memory.used,memory.total,power.draw")
        .arg("--format=csv,noheader,nounits")
        .output();

    match output {
        Ok(output) => {
            let mut gpu_model = String::new();
            let mut temps = Vec::new();
            let mut vram_used = 0;
            let mut vram_total = 0;
            let mut total_power = 0.0;
            let mut gpu_count = 0;

            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                if parts.len() == 5 {
                    if gpu_model.is_empty() {
                        gpu_model = parts[0].replace("NVIDIA GeForce ", "");
                    }
                    temps.push(parts[1].to_string());
                    vram_used += parts[2].parse::<i32>().unwrap_or(0);
                    vram_total += parts[3].parse::<i32>().unwrap_or(0);
                    total_power += parts[4].parse::<f32>().unwrap_or(0.0);
                    gpu_count += 1;
                }
            }

            if gpu_count == 0 {
                return "Failed to parse GPU stats".to_string();
            }

            format!(
                "{}x {} | Temp: {}°C | VRAM: {} MB / {} MB | Power: {:.2} W",
                gpu_count,
                gpu_model,
                temps.join(", "),
                vram_used,
                vram_total,
                total_power
            )
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
        chat_history: Arc::new(Mutex::new(Vec::new())),
        system_prompt: Arc::new(Mutex::new(SYSTEM_PROMPT.to_string())),
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
                task::spawn(monitor_gpu_stats(ctx_clone));

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
