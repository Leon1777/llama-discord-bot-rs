use llama_cpp_2::{
    context::params::LlamaContextParams,
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
};
use std::{
    num::NonZeroU32,
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub async fn generate_response(
    user_input: &str,
    chat_history: Arc<Mutex<Vec<Message>>>,
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
) -> Result<String, String> {
    // update global chat history with user input
    {
        let mut history = chat_history.lock().unwrap();
        history.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        if history.len() > 5 {
            let excess = history.len() - 5;
            history.drain(0..excess); // drain old messages
        }
    }

    // local copy of the chat history
    let local_history = {
        let history = chat_history.lock().unwrap();
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

    // println!("Tokenized output: {:?}", tokens_list);

    let tokens_in_history = tokens_list.len() as i32;
    let n_ctx = 32768; // input + output tokens

    // n_len is the max tokens the model can generate
    // [n_batch >= tokens_in_history + n_len]
    //
    // TODO: Improve dynamic adjustment based on
    // input/tokens_in_history or pass via !ask
    let n_len = 1024;

    // n_batch is the number of tokens processed at once
    // It must be at least tokens_in_history + n_len to avoid a "batch size exceeded" error
    //
    // TODO: verify if this method is optimal for calculating n_batch
    let n_batch = std::cmp::min(tokens_in_history + n_len, 4096) as usize;
    println!(
        "Total tokens required: {}, Input tokens: {}, Calculated n_batch: {}",
        tokens_in_history + n_len,
        tokens_in_history,
        n_batch
    );

    // context parameters
    // - n_ctx sets the total context size (input + output tokens)
    // - n_batch ensures the batch can handle all input and generated tokens
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx as u32))
        .with_n_batch(n_batch as u32)
        .with_n_threads(6);
    // If your token generation is extremely slow, try setting this number
    // to 1. If this significantly improves your token generation speed,
    // then your CPU is being oversaturated and you need to explicitly set
    // this parameter to the number of the physical CPU cores on your machine
    // (even if you utilize a GPU). If in doubt, start with 1 and double
    // the amount until you hit a performance bottleneck, then scale the number down.

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| format!("Failed to create context: {}", e))?;

    let mut batch = LlamaBatch::new(n_batch, 1);
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

    let mut sampler = LlamaSampler::chain_simple([
        // LlamaSampler::temp(0.7),                    // Control randomness
        // LlamaSampler::top_k(50),                    // Top-k sampling for diversity
        // LlamaSampler::top_p(0.9, 1),                // Nucleus sampling for quality
        // LlamaSampler::penalties(64, 1.2, 0.0, 0.0), // Penalize repetition
        // LlamaSampler::greedy(),                     // Fallback to highest-probability token
        LlamaSampler::mirostat_v2(42, 5.0, 0.1), // Mirostat v2 for adaptive generation
    ]);

    // Mirostat 2.0 algorithm described in the paper <https://arxiv.org/abs/2007.14966>. Uses tokens instead of words.
    //
    // # Parameters:
    // - ``seed``: Seed to initialize random generation with.
    // - ``tau``: The target cross-entropy (or surprise) value you want to achieve for the
    //     generated text. A higher value corresponds to more surprising or less predictable text,
    //     while a lower value corresponds to less surprising or more predictable text.
    // - ``eta``: The learning rate used to update `mu` based on the error between the target and
    //     observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
    //     updated more quickly, while a smaller learning rate will result in slower updates.

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

    // user input and response to global chat history only if the generation was successful
    if !generated_response.is_empty() {
        let mut history = chat_history.lock().unwrap();
        history.push(Message {
            role: "assistant".to_string(),
            content: generated_response.clone(),
        });

        println!("Chat History: {:?}", history);
    }

    Ok(generated_response)
}
