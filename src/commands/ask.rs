use crate::bot::inference::generate_response;
use crate::bot::utils::{process_links, split_message};
use crate::BotContext;
use crate::BotError;
use poise::CreateReply;
use std::sync::Arc;

#[poise::command(slash_command, track_edits)]
pub async fn ask(
    ctx: poise::Context<'_, Arc<BotContext>, BotError>,
    #[description = "Ask the AI anything"] question: String,
) -> Result<(), BotError> {
    let state = ctx.data();

    let _lock = state.request_lock.lock().await;

    ctx.defer().await?;

    // handle links
    let processed_question = process_links(&question)
        .await
        .unwrap_or_else(|_| question.clone());
    println!("Processed Prompt: {}", processed_question);

    let response = generate_response(
        &processed_question,
        Arc::clone(&state.chat_history),
        Arc::clone(&state.model),
        Arc::clone(&state.backend),
    )
    .await
    .unwrap_or_else(|e| {
        eprintln!("Error generating response: {}", e);
        "Error generating response.".to_string()
    });

    let full_response = format!(
        "**{} asked:** {}\n\n**AI Response:** {}",
        ctx.author().name,
        question,
        response
    );

    let messages: Vec<String> = split_message(&full_response, 2000);
    let mut messages_iter = messages.into_iter();

    if let Some(first_message) = messages_iter.next() {
        ctx.send(CreateReply::default().content(first_message))
            .await?;
    }

    for message in messages_iter {
        ctx.channel_id()
            .say(&ctx.serenity_context().http, message)
            .await?;
    }
    Ok(())
}
