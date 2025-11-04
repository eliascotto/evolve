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
    TypeError(String),
    ValueError(String),
    IndexError(String),
    KeyError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::SyntaxError(e) => write!(f, "{}", e),
            Error::RuntimeError(e) => write!(f, "Runtime error: {}", e),
            Error::TypeError(e) => write!(f, "Type error: {}", e),
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
            SyntaxError::InvalidMap { reason } => write!(f, "Invalid map: {}", reason),
        }
    }
}

//===----------------------------------------------------------------------===//
// ErrorWithSpan
//===----------------------------------------------------------------------===//

#[derive(Debug, Clone)]
pub struct ErrorWithSpan {
    pub error: Error,
    pub span: Span,
    pub source: String, // Keep original source for context
    pub source_location: Source,
}

impl ErrorWithSpan {
    pub fn format_error(&self) -> String {
        let line_start =
            self.source[..self.span.start].rfind('\n').map(|pos| pos + 1).unwrap_or(0);

        let line_end = self.source[self.span.start..]
            .find('\n')
            .map(|pos| self.span.start + pos)
            .unwrap_or(self.source.len());

        let line_number = self.source[..self.span.start].matches('\n').count() + 1;
        let column = self.span.start - line_start + 1;

        let line_content = &self.source[line_start..line_end];
        let underline = " ".repeat(column - 1) + &"^".repeat(self.span.len());

        format!(
            "Error at ({} {}:{})\n{}\n{}\n{}",
            self.source_location.display(),
            line_number,
            column,
            line_content,
            underline,
            self.error.to_string()
        )
    }
}
