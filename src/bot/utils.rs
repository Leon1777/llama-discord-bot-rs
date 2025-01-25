pub fn split_message(content: &str, max_length: usize) -> Vec<String> {
    content
        .chars()
        .collect::<Vec<_>>()
        .chunks(max_length)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect()
}
