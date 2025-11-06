use logos::Span;
use std::fmt;

use crate::reader::Source;

//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//

#[derive(Debug, Clone)]
pub enum Error {
    SyntaxError(SyntaxError),
    RuntimeError(String),
    TypeError(String, String),
    ValueError(String),
    IndexError(String),
    KeyError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::SyntaxError(e) => write!(f, "{}", e),
            Error::RuntimeError(e) => write!(f, "Runtime error: {}", e),
            Error::TypeError(expected, got) => {
                write!(f, "Type error: expected {}, got {}", expected, got)
            }
            Error::ValueError(e) => write!(f, "Value error: {}", e),
            Error::IndexError(e) => write!(f, "Index error: {}", e),
            Error::KeyError(e) => write!(f, "Key error: {}", e),
        }
    }
}

//===----------------------------------------------------------------------===//
// SyntaxError
//===----------------------------------------------------------------------===//

#[derive(Debug, Clone)]
pub enum SyntaxError {
    UnexpectedEOF { expected: Option<String> },
    UnexpectedToken { found: String, expected: String },
    UnbalancedDelimiter { delimiter: char, position: usize },
    BadEscape { sequence: String },
    UnterminatedString,
    InvalidNumber { value: String },
    InvalidCharacter { char: char },
    InvalidKeyword { value: String },
    InvalidSymbol { value: String },
    InvalidList { reason: String },
    InvalidVector { reason: String },
    InvalidMap { reason: String },
    InvalidMeta { reason: String },
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SyntaxError::UnexpectedEOF { expected } => {
                write!(
                    f,
                    "Unexpected EOF: {}",
                    expected.clone().unwrap_or("None".to_string())
                )
            }
            SyntaxError::UnexpectedToken { found, expected } => {
                write!(f, "Unexpected token: {} (expected: {})", found, expected)
            }
            SyntaxError::UnbalancedDelimiter { delimiter, position } => {
                write!(
                    f,
                    "Unbalanced delimiter: {} at position {}",
                    delimiter, position
                )
            }
            SyntaxError::BadEscape { sequence } => {
                write!(f, "Bad escape sequence: {}", sequence)
            }
            SyntaxError::UnterminatedString => write!(f, "Unterminated string"),
            SyntaxError::InvalidNumber { value } => {
                write!(f, "Invalid number: {}", value)
            }
            SyntaxError::InvalidCharacter { char } => {
                write!(f, "Invalid character: {}", char)
            }
            SyntaxError::InvalidKeyword { value } => {
                write!(f, "Invalid keyword: {}", value)
            }
            SyntaxError::InvalidSymbol { value } => {
                write!(f, "Invalid symbol: {}", value)
            }
            SyntaxError::InvalidList { reason } => {
                write!(f, "Invalid list: {}", reason)
            }
            SyntaxError::InvalidVector { reason } => {
                write!(f, "Invalid vector: {}", reason)
            }
            SyntaxError::InvalidMap { reason } => {
                write!(f, "Invalid map: {}", reason)
            }
            SyntaxError::InvalidMeta { reason } => {
                write!(f, "{}", reason)
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// ErrorWithSpan
//===----------------------------------------------------------------------===//

/// A wrapper around an error that includes a span, source, file, and optional secondary spans and notes.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// The error that occurred (syntax, runtime, type, value, index, or key error).
    pub error: Error,
    /// The primary span indicating where the error occurred in the source code.
    /// This span defines the byte range (start, end) that marks the problematic location.
    pub span: Span,
    /// The complete source code string from which the error originated.
    /// Used to extract context lines, display code snippets, and calculate line/column positions.
    pub source: String,
    /// The source file identifier (file path or REPL input) where the error occurred.
    pub file: Source,
    /// Optional secondary spans marking related locations in the source code.
    /// Commonly used to highlight the opening delimiter when a closing delimiter is unmatched,
    /// or to show other related error positions that provide additional context.
    pub secondary_spans: Option<Vec<Span>>,
    /// Optional additional explanatory notes to display with the error message.
    /// These notes provide helpful hints, suggestions, or context about the error
    /// that can assist users in understanding and fixing the issue.
    pub notes: Option<Vec<String>>,
}

impl Diagnostic {
    pub fn new(error: Error, span: Span, source: String, file: Source) -> Self {
        Self { error, span, source, file, secondary_spans: None, notes: None }
    }

    /// Calculate line and column information for a span
    fn location_info(&self, span: &Span) -> (usize, usize, usize, usize) {
        // Find line start
        let line_start =
            self.source[..span.start].rfind('\n').map(|pos| pos + 1).unwrap_or(0);

        // Find line end
        let line_end = self.source[span.start..]
            .find('\n')
            .map(|pos| span.start + pos)
            .unwrap_or(self.source.len());

        let line_number = self.source[..span.start].matches('\n').count() + 1;
        let column = span.start - line_start + 1;

        (line_number, column, line_start, line_end)
    }

    /// Get context lines around the error (previous and next lines)
    fn get_context_lines(
        &self,
        target_line: usize,
        line_start: usize,
    ) -> Vec<(usize, String)> {
        let mut context = Vec::new();

        // Previous line
        if target_line > 1 {
            let prev_line_start = self.source[..line_start]
                .rfind('\n')
                .map(|pos| pos + 1)
                .unwrap_or(0);
            if prev_line_start < line_start {
                let prev_line_end = line_start - 1;
                let prev_line_content = &self.source[prev_line_start..prev_line_end];
                context.push((target_line - 1, prev_line_content.to_string()));
            }
        }

        // Current line is handled separately
        // Next line
        let next_line_start = self.source[line_start..]
            .find('\n')
            .map(|pos| line_start + pos + 1)
            .unwrap_or(self.source.len());

        if next_line_start < self.source.len() {
            let next_line_end = self.source[next_line_start..]
                .find('\n')
                .map(|pos| next_line_start + pos)
                .unwrap_or(self.source.len());

            if next_line_end > next_line_start {
                let next_line_content = &self.source[next_line_start..next_line_end];
                context.push((target_line + 1, next_line_content.to_string()));
            }
        }

        context
    }

    /// Format a single line with its number
    fn format_line(&self, line_num: usize, content: &str) -> String {
        format!("{:4} | {}", line_num, content)
    }

    /// Format the underline/caret for a span on a line
    fn format_underline(&self, column: usize, span_len: usize) -> String {
        let padding = " ".repeat(column - 1);
        let caret_len = span_len.max(1);
        let caret = "^".repeat(caret_len);
        format!("     | {}{}", padding, caret)
    }

    /// Formats a comprehensive, multi-line error message with contextual information.
    ///
    /// Returns a formatted error string that includes:
    /// - The error message
    /// - File location (file path, line number, and column)
    /// - Source code context (surrounding lines when available)
    /// - Visual indicators (carets) pointing to the error location
    /// - Secondary spans (e.g., related delimiter positions)
    /// - Additional notes, if provided
    ///
    /// The output format is similar to Rust compiler error messages, making it
    /// easy to identify and fix issues in source code.
    pub fn format(&self) -> String {
        let (line_num, column, line_start, line_end) =
            self.location_info(&self.span);

        let line_content = &self.source[line_start..line_end];

        let mut output = String::new();

        // Error message
        output.push_str(&format!("{}\n", self.error));

        // Location header
        output.push_str(&format!(
            "  --> {}:{}:{}\n",
            self.file.display(),
            line_num,
            column
        ));

        output.push_str("   |\n");

        // Context: previous line if available
        let context_lines = self.get_context_lines(line_num, line_start);
        for (ctx_line_num, ctx_content) in &context_lines {
            if *ctx_line_num < line_num {
                output.push_str(&self.format_line(*ctx_line_num, ctx_content));
                output.push_str("\n");
            }
        }

        // Primary error line
        output.push_str(&self.format_line(line_num, line_content));
        output.push_str("\n");

        // Primary underline
        output.push_str(&self.format_underline(column, self.span.len()));
        output.push_str("\n");

        // Secondary spans (e.g., opening delimiter for unmatched closing)
        if let Some(secondary_spans) = &self.secondary_spans {
            for sec_span in secondary_spans {
                let (sec_line, sec_col, sec_start, sec_end) =
                    self.location_info(sec_span);
                let sec_content = &self.source[sec_start..sec_end];

                if sec_line == line_num {
                    // Same line - show on same underline
                    let sec_padding = " ".repeat(sec_col - 1);
                    let sec_caret = "^".repeat(sec_span.len().max(1));
                    output.push_str(&format!(
                        "     | {}{} (opening delimiter here)\n",
                        sec_padding, sec_caret
                    ));
                } else {
                    // Different line - show separately
                    output.push_str("   |\n");
                    output.push_str(&self.format_line(sec_line, sec_content));
                    output.push_str("\n");
                    output.push_str(&self.format_underline(sec_col, sec_span.len()));
                    output.push_str(" (opening delimiter)\n");
                }
            }
        }

        // Notes
        if let Some(notes) = &self.notes {
            output.push_str("   |\n");
            for note in notes {
                output.push_str(&format!("   = note: {}\n", note));
            }
        }

        output
    }
}
