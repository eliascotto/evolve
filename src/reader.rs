use logos::{Logos, Span};
use std::{fmt, fs, path};

use crate::error::{Diagnostic, Error, SyntaxError};
use crate::interner;
use crate::value::{self, Value};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Unescapes a string literal by converting escape sequences to their actual characters.
/// Handles common escape sequences: \n, \t, \r, \", \\, and others.
fn unescape_string(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next_ch) = chars.next() {
                match next_ch {
                    'n' => result.push('\n'),
                    't' => result.push('\t'),
                    'r' => result.push('\r'),
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    '0' => result.push('\0'),
                    _ => {
                        // For unknown escape sequences, keep the backslash and character
                        result.push('\\');
                        result.push(next_ch);
                    }
                }
            } else {
                // Trailing backslash, keep it
                result.push('\\');
            }
        } else {
            result.push(ch);
        }
    }

    result
}

//===----------------------------------------------------------------------===//
// Source
//===----------------------------------------------------------------------===//
#[derive(Debug, PartialEq, Clone)]
pub enum Source {
    File(path::PathBuf),
    REPL,
}

impl Source {
    pub fn display(&self) -> String {
        match self {
            Source::File(path) => path.display().to_string(),
            Source::REPL => "REPL".to_string(),
        }
    }
}

//===----------------------------------------------------------------------===//
// Token
//
// Uses logos crate to implement the tokenizer, bringing fast and efficient
// tokenization and parsing without the need for regular expressions.
//===----------------------------------------------------------------------===//

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // --------- Skips ---------
    // Whitespace and commas are ignored (they were outside your capture group).
    #[regex(r"[ \t\r\n,]+", logos::skip)]
    // Line comments: from ; to end-of-line.
    #[regex(r";[^\n]*", logos::skip)]
    // --------------------------------

    // --------- Delimiters & Punct ---------
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    // Quote-ish/reader macros
    #[token("'")]
    Quote,
    #[token("`")]
    Backtick,
    #[token("^")]
    Caret,
    #[token("@")]
    At,
    #[token("#")]
    Hash,
    #[token("_")]
    Underscore,
    #[token("~@")]
    TildeAt,
    #[token("~")]
    Tilde,

    // Comments
    #[regex(r";[^\n]*", priority = 0)]
    Comment,

    // --------- Characters ---------
    #[regex(r"'\\.'",
      priority = 2,
      callback = |lex| lex.slice().parse::<char>().unwrap())]
    Char(char),

    // --------- Literals ---------
    // String: accept \" and \\ and any escaped char.
    // Your original allowed possibly unterminated (note the trailing "?").
    // Here we split into two: a proper string and an unterminated one -> Error.
    #[regex(r#""([^"\\]|\\.)*""#,
      callback = |lex| {
        let slice = lex.slice();
        // Remove the surrounding quotes and unescape the content
        let content = &slice[1..slice.len()-1];
        unescape_string(content)
      })]
    Str(String),

    // Unterminated string: starts with " and runs to EOF without a closing ".
    // Logos will pick the earlier matching rule; keep this AFTER proper Str.
    #[regex(r#""([^"\\]|\\.)*"#,
      priority = 0,
      callback = |lex| lex.slice().to_owned())]
    UnterminatedStr(String),

    // Numbers — mirror your regexes:
    // ints: ^\d+$
    #[regex(r"-?\d+",
      priority = 4,
      callback = |lex| lex.slice().parse::<i64>().unwrap())]
    Int(i64),

    // floats: ^-?\d+(\.\d+)?$
    // (Tokenization only: leading '-' is included here to match your pattern)
    #[regex(r"-?\d+(?:\.\d+)?",
      priority = 3,
      callback = |lex| lex.slice().parse::<f64>().unwrap())]
    Float(f64),

    // --------- Symbols / Idents ---------
    // Everything else that your final alt matched:
    // [^\s\[\]{}('"`,;^@#_~)]+
    #[regex(r#"[^ \t\r\n\[\]{}\('"`,;^@#_~)]+"#,
      priority = 0,
      callback = |lex| lex.slice().to_owned())]
    Symbol(String),

    #[regex(r#":[^ \t\r\n\[\]{}\('"`,;^@#_~)]+"#,
      priority = 1,
      callback = |lex| lex.slice().to_owned())]
    Keyword(String),
}

/// Displays a Token as a string.
/// Used for debugging purposes and proper error message formatting.
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::LParen => write!(f, r"("),
            Token::RParen => write!(f, r")"),
            Token::LBracket => write!(f, r"["),
            Token::RBracket => write!(f, r"]"),
            Token::LBrace => write!(f, r"{{"),
            Token::RBrace => write!(f, r"}}"),
            Token::Quote => write!(f, "'"),
            Token::Backtick => write!(f, "`"),
            Token::Caret => write!(f, "^"),
            Token::At => write!(f, "@"),
            Token::Hash => write!(f, "#"),
            Token::Underscore => write!(f, "_"),
            Token::TildeAt => write!(f, "~@"),
            Token::Tilde => write!(f, "~"),
            Token::Comment => write!(f, ";"),
            Token::Char(char) => write!(f, r"{}", char),
            Token::Str(str) => write!(f, r"{}", str),
            Token::UnterminatedStr(str) => write!(f, r"{}", str),
            Token::Int(int) => write!(f, r"{}", int),
            Token::Float(float) => write!(f, r"{}", float),
            Token::Symbol(symbol) => write!(f, r"{}", symbol),
            Token::Keyword(keyword) => write!(f, r"{}", keyword),
        }
    }
}

/// A wrapper around a Token and its span.
/// Stores span and source location, for error reporting.
#[derive(Debug, PartialEq, Clone)]
pub struct TokenCST {
    token: Token,
    span: Span,
    source: Source,
}

#[derive(Debug)]
pub struct Reader {
    tokens: Vec<TokenCST>,
    source: String,
    position: usize,
    file: Source,
}

impl Reader {
    /// Reads the next token from the reader and returns a TokenAST. Increments the position.
    pub fn next(&mut self) -> Result<&TokenCST, Diagnostic> {
        let token = match self.tokens.get(self.position) {
            Some(t) => t,
            None => {
                return Err(Diagnostic {
                    error: Error::SyntaxError(SyntaxError::UnexpectedEOF {
                        expected: None,
                    }),
                    span: self.last_span(),
                    source: self.source.clone(),
                    file: self.file.clone(),
                    secondary_spans: None,
                    notes: None,
                });
            }
        };

        self.position += 1;
        Ok(token)
    }

    /// Peeks the next token from the reader and returns a TokenAST. Does not increment the position.
    pub fn peek(&self) -> Result<&TokenCST, Diagnostic> {
        match self.tokens.get(self.position) {
            Some(t) => Ok(t),
            None => Err(Diagnostic {
                error: Error::SyntaxError(SyntaxError::UnexpectedEOF {
                    expected: None,
                }),
                span: self.last_span(),
                source: self.source.clone(),
                file: self.file.clone(),
                secondary_spans: None,
                notes: None,
            }),
        }
    }

    /// Returns the span of the last token in the reader.
    fn last_span(&self) -> Span {
        self.tokens.last().unwrap().span.clone()
    }

    /// Returns the span of the current token in the reader.
    fn current_span(&self) -> Span {
        self.tokens.get(self.position).unwrap().span.clone()
    }
}

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

pub fn tokenize(source_code: &str, source_location: Source) -> Reader {
    let mut lexer = Token::lexer(source_code.trim());
    let mut tokens: Vec<TokenCST> = vec![];

    while let Some(token) = lexer.next() {
        if let Ok(token) = token {
            match token {
                Token::Comment => continue,
                _ => tokens.push(TokenCST {
                    token,
                    span: lexer.span(),
                    source: source_location.clone(),
                }),
            }
        }
    }

    Reader {
        tokens,
        source: source_code.to_string(),
        position: 0,
        file: source_location,
    }
}

//===----------------------------------------------------------------------===//
// Reader
//===----------------------------------------------------------------------===//

/// Reads an atom from the reader and returns a Value.
/// An atom is a single value that is not a collection.
/// It can be a symbol, a number, a string, a character, or a keyword.
fn read_atom(reader: &mut Reader) -> Result<Value, Diagnostic> {
    let token_ast = reader.next()?;
    match &token_ast.token {
        Token::Symbol(symbol) => match symbol.as_str() {
            "nil" => Ok(Value::Nil { span: token_ast.span.clone() }),
            "true" => Ok(Value::Bool { span: token_ast.span.clone(), value: true }),
            "false" => {
                Ok(Value::Bool { span: token_ast.span.clone(), value: false })
            }
            _ => {
                let sym_id = interner::intern_sym(symbol.as_str());
                Ok(Value::Symbol {
                    span: token_ast.span.clone(),
                    value: sym_id,
                    meta: None,
                })
            }
        },
        Token::Keyword(keyword) => {
            let kw_id = interner::intern_kw(keyword.as_str());
            Ok(Value::Keyword { span: token_ast.span.clone(), value: kw_id })
        }
        Token::Int(int) => {
            Ok(Value::Int { span: token_ast.span.clone(), value: int.clone() })
        }
        Token::Float(float) => {
            Ok(Value::Float { span: token_ast.span.clone(), value: float.clone() })
        }
        Token::Str(str) => {
            Ok(Value::String { span: token_ast.span.clone(), value: str.clone() })
        }
        Token::Char(char) => {
            Ok(Value::Char { span: token_ast.span.clone(), value: char.clone() })
        }
        Token::UnterminatedStr(_) => Err(Diagnostic {
            error: Error::SyntaxError(SyntaxError::UnterminatedString),
            span: token_ast.span.clone(),
            source: reader.source.clone(),
            file: reader.file.clone(),
            secondary_spans: None,
            notes: Some(vec![
                "add a closing `\"` to terminate the string".to_string(),
            ]),
        }),
        _ => Err(Diagnostic {
            error: Error::SyntaxError(SyntaxError::UnexpectedToken {
                found: format!("{}", token_ast.token),
                expected: "atom".to_string(),
            }),
            span: token_ast.span.clone(),
            source: reader.source.clone(),
            file: reader.file.clone(),
            secondary_spans: None,
            notes: None,
        }),
    }
}

#[derive(Debug, PartialEq, Clone)]
enum CollType {
    List,
    Vector,
    HashMap,
    Set,
}

/// Reads a sequence of tokens and returns a Value of the given collection type.
fn read_sequence(
    reader: &mut Reader,
    coll_type: CollType,
) -> Result<Value, Diagnostic> {
    let open_span = reader.next()?.span.clone();
    let mut seq: Vec<Value> = vec![];

    loop {
        let token = match reader.peek() {
            Ok(t) => t.token.clone(),
            Err(_) => {
                return Err(Diagnostic {
                    error: Error::SyntaxError(SyntaxError::UnbalancedDelimiter {
                        delimiter: match coll_type {
                            CollType::List => '(',
                            CollType::Vector => '[',
                            CollType::HashMap | CollType::Set => '{',
                        },
                        position: open_span.start,
                    }),
                    span: reader.current_span(), // Where EOF/unexpected token is
                    source: reader.source.clone(),
                    file: reader.file.clone(),
                    secondary_spans: Some(vec![open_span.clone()]), // Show where the opening delimiter is
                    notes: None,
                });
            }
        };
        match token {
            Token::RParen if coll_type == CollType::List => break,
            Token::RBracket if coll_type == CollType::Vector => break,
            Token::RBrace
                if matches!(coll_type, CollType::HashMap | CollType::Set) =>
            {
                break;
            }
            _ => {
                seq.push(read_form(reader)?);
            }
        }
    }

    let close_span = reader.next()?.span.clone(); // consume closing delimiter
    // Span should cover the entire sequence, including the opening and closing delimiters
    let seq_span = Span { start: open_span.start, end: close_span.end };
    match coll_type {
        CollType::List => Ok(Value::List { span: seq_span, value: seq, meta: None }),
        CollType::Vector => {
            Ok(Value::Vector { span: seq_span, value: seq, meta: None })
        }
        CollType::HashMap => Ok(Value::Map {
            span: seq_span,
            value: value::create_btree_map_from_sequence(seq),
            meta: None,
        }),
        CollType::Set => Ok(Value::Set {
            span: seq_span,
            value: value::create_btree_set_from_sequence(seq),
            meta: None,
        }),
    }
}

/// Reads a single form from the reader and returns a Value.
fn read_form(reader: &mut Reader) -> Result<Value, Diagnostic> {
    let token_ast = reader.peek()?;
    match token_ast.token {
        // --------- Quote ---------
        Token::Quote => {
            let quote_span = token_ast.span.clone();
            let _ = reader.next()?;
            Ok(Value::List {
                span: quote_span.clone(),
                value: vec![
                    Value::Symbol {
                        span: quote_span,
                        value: interner::intern_sym("quote"),
                        meta: None,
                    },
                    read_form(reader)?,
                ],
                meta: None,
            })
        }

        // --------- ~ ---------
        Token::Tilde => {
            let tilde_span = token_ast.span.clone();
            let _ = reader.next()?;
            Ok(Value::List {
                span: tilde_span.clone(),
                value: vec![
                    Value::Symbol {
                        span: tilde_span.clone(),
                        value: interner::intern_sym("unquote"),
                        meta: None,
                    },
                    read_form(reader)?,
                ],
                meta: None,
            })
        }

        // --------- ~@ ---------
        Token::TildeAt => {
            let tilde_at_span = token_ast.span.clone();
            let _ = reader.next()?;
            Ok(Value::List {
                span: tilde_at_span.clone(),
                value: vec![
                    Value::Symbol {
                        span: tilde_at_span.clone(),
                        value: interner::intern_sym("unquote-splicing"),
                        meta: None,
                    },
                    read_form(reader)?,
                ],
                meta: None,
            })
        }

        // --------- ^ ---------
        Token::Caret => {
            let caret_span = token_ast.span.clone();
            let _ = reader.next()?; // consume the caret
            let meta = read_form(reader)?;
            let meta_span = match &meta {
                Value::Keyword { span, .. } => span.clone(),
                Value::Map { span, .. } => span.clone(),
                _ => caret_span.clone(),
            };

            let next_object = read_form(reader)?;
            let object_with_meta =
                next_object.set_meta(meta).map_err(|e| Diagnostic {
                    error: e,
                    span: Span { start: caret_span.start, end: meta_span.end },
                    source: reader.source.clone(),
                    file: reader.file.clone(),
                    secondary_spans: None,
                    notes: None,
                })?;
            Ok(object_with_meta)
        }

        // --------- # ---------
        Token::Hash => {
            let _ = reader.next()?; // remove the hash
            let next_token_ast = reader.peek()?;
            match next_token_ast.token {
                // Var-quote (#'sym)
                Token::Quote => {
                    let var_quote_span = next_token_ast.span.clone();
                    let _ = reader.next()?;
                    Ok(Value::List {
                        span: var_quote_span.clone(),
                        value: vec![
                            Value::Symbol {
                                span: var_quote_span,
                                value: interner::intern_sym("var"),
                                meta: None,
                            },
                            read_form(reader)?,
                        ],
                        meta: None,
                    })
                }
                // Ignore next form (#_form)
                Token::Underscore => {
                    let _ = reader.next()?;
                    let _ = read_form(reader)?; // discard next form
                    read_form(reader)
                }
                // Create a set (#{..})
                Token::LBrace => read_sequence(reader, CollType::Set),
                _ => {
                    return Err(Diagnostic {
                        error: Error::SyntaxError(SyntaxError::UnexpectedToken {
                            found: format!("{:?}", next_token_ast.token),
                            expected: "set".to_string(),
                        }),
                        span: next_token_ast.span.clone(),
                        source: reader.source.clone(),
                        file: reader.file.clone(),
                        secondary_spans: None,
                        notes: None,
                    });
                }
            }
        }

        // --------- Sequences ---------
        Token::LParen => read_sequence(reader, CollType::List),
        Token::LBracket => read_sequence(reader, CollType::Vector),
        Token::LBrace => read_sequence(reader, CollType::HashMap),
        _ => read_atom(reader),
    }
}

pub fn read_file(path: &path::Path) -> Result<Value, Diagnostic> {
    let source_code = fs::read_to_string(path).map_err(|e| Diagnostic {
        error: Error::RuntimeError(format!("Failed to read file: {}", e)),
        span: logos::Span { start: 0, end: 0 },
        source: String::new(),
        file: Source::File(path.to_path_buf()),
        secondary_spans: None,
        notes: None,
    })?;
    let mut reader = tokenize(&source_code, Source::File(path.to_path_buf()));
    read_form(&mut reader)
}

pub fn read(source: &str) -> Result<Value, Diagnostic> {
    let mut reader = tokenize(source, Source::REPL);
    read_form(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_core_cases() {
        let src: &str = r#"(def x 123 -45.6 "hi\n\"there\"" ; comment
                      ~@ [a b] {k v} 'sym `q ^m @n #t _ ~ sym-1 \a)"#;

        let toks: Vec<TokenCST> = tokenize(src, Source::REPL).tokens;

        // Smoke-check presence/order of a few key tokens:
        assert!(toks.starts_with(&[TokenCST {
            token: Token::LParen,
            span: Span { start: 0, end: 1 },
            source: Source::REPL,
        }]));
        assert!(toks.contains(&TokenCST {
            token: Token::Int(123),
            span: Span { start: 7, end: 10 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Float(-45.6),
            span: Span { start: 11, end: 16 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Str("hi\n\"there\"".to_string()),
            span: Span { start: 17, end: 32 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::TildeAt,
            span: Span { start: 65, end: 67 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::LBracket,
            span: Span { start: 68, end: 69 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::LBrace,
            span: Span { start: 74, end: 75 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Quote,
            span: Span { start: 80, end: 81 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Backtick,
            span: Span { start: 85, end: 86 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Caret,
            span: Span { start: 88, end: 89 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::At,
            span: Span { start: 91, end: 92 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Hash,
            span: Span { start: 94, end: 95 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Underscore,
            span: Span { start: 97, end: 98 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Tilde,
            span: Span { start: 99, end: 100 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Symbol("sym-1".to_string()),
            span: Span { start: 101, end: 106 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Symbol("\\a".to_string()),
            span: Span { start: 107, end: 109 },
            source: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::RParen,
            span: Span { start: 109, end: 110 },
            source: Source::REPL,
        }));
    }

    #[test]
    fn unterminated_string_is_flagged() {
        let src = r#""oops"#;
        let v: Vec<_> = tokenize(src, Source::REPL).tokens;
        assert_eq!(
            v,
            vec![TokenCST {
                token: Token::UnterminatedStr("\"oops".to_string()),
                span: Span { start: 0, end: 5 },
                source: Source::REPL,
            }]
        );
    }
}
