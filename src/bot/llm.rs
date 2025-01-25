use llama_cpp_2::{
    context::params::LlamaContextParams,
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::LlamaModel,
    model::{AddBos, Special},
    sampling::LlamaSampler,
};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub async fn generate_response(
    user_input: &str,
    chat_history: Arc<Mutex<Vec<Message>>>,
    n_len: i32,
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
) -> Result<String, String> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
        .with_n_threads(14); // threads for CPU decoding

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| format!("Failed to create context: {}", e))?;

    // local copy of chat history
    let local_history = {
        let mut history = chat_history.lock().unwrap();

        history.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // separate scope
        if history.len() > 5 {
            let excess = history.len() - 5;
            history.drain(0..excess); // drain old messages
        }

        // for local use
        history.clone()
    };

    // construct prompt from local history
    let prompt = local_history
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
        + "\nAssistant:";

    // tokenize prompt
    let tokens_list = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| format!("Failed to tokenize prompt: {}", e))?;

    println!("Tokens for prompt: {:?}", tokens_list);

    let n_ctx = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

    if n_kv_req > n_ctx {
        return Err(format!(
            "KV cache size is insufficient: n_kv_req={} > n_ctx={}. Reduce n_len or increase n_ctx.",
            n_kv_req, n_ctx
        ));
    }

    // sampler
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.7),                    // Control randomness
        LlamaSampler::top_k(50),                    // Top-k sampling for diversity
        LlamaSampler::top_p(0.9, 1),                // Nucleus sampling for quality
        LlamaSampler::penalties(64, 1.2, 0.0, 0.0), // Penalize repetition
        LlamaSampler::greedy(),                     // Fallback to highest-probability token
    ]);

    let mut batch = LlamaBatch::new(2048, 1);
    let last_index: i32 = (tokens_list.len() - 1) as i32;

    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_index;
        batch
            .add(token, i, &[0], is_last)
            .map_err(|e| format!("Failed to add token to batch: {}", e))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| format!("Failed to decode batch: {}", e))?;

    // decode main loop
    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;
    let t_main_start = ggml_time_us();

    let mut generated_response = String::new();

    while n_cur < n_len {
        if n_cur <= 0 {
            return Err("No valid tokens for sampling.".to_string());
        }

        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        // token to string
        if let Ok(token_str) = model.token_to_str(token, Special::Tokenize) {
            generated_response += &token_str;
        } else {
            eprintln!("Skipping invalid token: {}", token);
        }

        // Prepare for next token
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| format!("Failed to add token to batch: {}", e))?;

        ctx.decode(&mut batch)
            .map_err(|e| format!("Failed to decode batch: {}", e))?;

        n_cur += 1;
        n_decode += 1;
    }

    let t_main_end = ggml_time_us();
    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

    println!(
        "Decoded {} tokens in {:.2}s, speed: {:.2} t/s",
        n_decode,
        duration.as_secs_f32(),
        n_decode as f32 / duration.as_secs_f32()
    );

    // Update global chat history with response
    {
        let mut history = chat_history.lock().unwrap();

        history.push(Message {
            role: "assistant".to_string(),
            content: generated_response.clone(),
        });

        println!("Chat History: {:?}", history);
    }

    Ok(generated_response)
}
