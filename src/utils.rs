/// Removes the last character from a string.
pub fn remove_last_char(s: String) -> String {
    s.chars().take(s.len() - 1).collect::<String>()
}

/// Creates a synthetic span with start and end set to 0.
/// This is useful for values that are generated programmatically rather than parsed.
#[macro_export]
macro_rules! synthetic_span {
    () => {
        crate::reader::Span { start: 0, end: 0 }
    };
}
