use regex::Regex;
use reqwest;
use scraper::{Html, Selector};
use std::{collections::HashSet, error::Error};

pub async fn fetch(url: &str) -> Result<String, Box<dyn Error>> {
    Ok(reqwest::get(url).await?.text().await?)
}

pub fn extract_text(html: &str) -> Result<String, Box<dyn Error>> {
    let sel = Selector::parse("title, h1, h2, h3, h4, h5, h6, p, li")?;
    let doc = Html::parse_document(html);

    let texts: HashSet<_> = doc.select(&sel).flat_map(|e| e.text().map(clean)).collect();

    Ok(texts.into_iter().collect::<Vec<_>>().join(" "))
}

/// clean whitespace
fn clean(text: &str) -> String {
    Regex::new(r"\s{2,}")
        .unwrap()
        .replace_all(text.trim(), " ")
        .to_string()
}

pub async fn process_links(prompt: &str) -> Result<String, Box<dyn Error>> {
    let re = Regex::new(r"https?://[^\s/$.?#].[^\s]*")?;
    let mut processed_prompt = prompt.to_string();

    for url in re.find_iter(prompt).map(|m| m.as_str()) {
        if let Ok(html) = fetch(url).await {
            if let Ok(text) = extract_text(&html) {
                // append extracted content after the URL
                processed_prompt =
                    processed_prompt.replace(url, &format!("{} (Content: {})", url, text));
            }
        }
    }

    Ok(processed_prompt)
}

pub fn split_message(content: &str, max_length: usize) -> Vec<String> {
    let mut result = Vec::new();
    let mut remaining = content;

    while !remaining.is_empty() {
        if remaining.len() <= max_length {
            result.push(remaining.to_string());
            break;
        }

        // find last whitespace within max_length
        let split_at = match remaining[..max_length].rfind(char::is_whitespace) {
            Some(pos) => pos,
            None => max_length, // split at max_length
        };

        let (part, rest) = remaining.split_at(split_at);

        result.push(part.trim_end().to_string());
        remaining = rest.trim_start();
    }

    result
}
