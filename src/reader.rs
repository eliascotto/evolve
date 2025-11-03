use logos::{Logos, Span};
use std::fmt;

use crate::error::{Error, ErrorWithSpan, SyntaxError};
use crate::list;
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
    #[regex(r"\d+",
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
/// Used for storing the token and its span in the reader, for error reporting.
#[derive(Debug, PartialEq, Clone)]
pub struct TokenAST {
    token: Token,
    span: Span,
}

#[derive(Debug)]
pub struct Reader {
    tokens: Vec<TokenAST>,
    source: String,
    position: usize,
}

impl Reader {
    /// Reads the next token from the reader and returns a TokenAST. Increments the position.
    pub fn next(&mut self) -> Result<&TokenAST, ErrorWithSpan> {
        let token = match self.tokens.get(self.position) {
            Some(t) => t,
            None => {
                return Err(ErrorWithSpan {
                    error: Error::SyntaxError(SyntaxError::UnexpectedEOF { expected: None }),
                    span: self.last_span(),
                    source: self.source.clone(),
                });
            }
        };

        self.position += 1;
        Ok(token)
    }

    /// Peeks the next token from the reader and returns a TokenAST. Does not increment the position.
    pub fn peek(&self) -> Result<&TokenAST, ErrorWithSpan> {
        match self.tokens.get(self.position) {
            Some(t) => Ok(t),
            None => Err(ErrorWithSpan {
                error: Error::SyntaxError(SyntaxError::UnexpectedEOF { expected: None }),
                span: self.last_span(),
                source: self.source.clone(),
            }),
        }
    }

    /// Returns the span of the last token in the reader.
    fn last_span(&self) -> Span {
        self.tokens.last().unwrap().span.clone()
    }
}

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

pub fn tokenize(source: &str) -> Reader {
    let mut lexer = Token::lexer(source.trim());
    let mut tokens: Vec<TokenAST> = vec![];

    while let Some(token) = lexer.next() {
        if let Ok(token) = token {
            match token {
                Token::Comment => continue,
                _ => tokens.push(TokenAST { token, span: lexer.span() }),
            }
        }
    }

    Reader { tokens, source: source.to_string(), position: 0 }
}

//===----------------------------------------------------------------------===//
// Reader
//===----------------------------------------------------------------------===//

/// Reads an atom from the reader and returns a Value.
/// An atom is a single value that is not a collection.
/// It can be a symbol, a number, a string, a character, or a keyword.
fn read_atom(reader: &mut Reader) -> Result<Value, ErrorWithSpan> {
    let token_ast = reader.next()?;
    match &token_ast.token {
        Token::Symbol(symbol) => match symbol.as_str() {
            "nil" => Ok(Value::Nil),
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            _ => Ok(Value::Symbol(symbol.clone())),
        },
        Token::Int(int) => Ok(Value::Int(int.clone())),
        Token::Float(float) => Ok(Value::Float(float.clone())),
        Token::Str(str) => Ok(Value::String(str.clone())),
        Token::Char(char) => Ok(Value::Char(char.clone())),
        Token::Keyword(keyword) => Ok(Value::Keyword(keyword.clone())),

        _ => Err(ErrorWithSpan {
            error: Error::SyntaxError(SyntaxError::UnexpectedToken {
                found: format!("{}", token_ast.token),
                expected: "atom".to_string(),
            }),
            span: token_ast.span.clone(),
            source: reader.source.clone(),
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
fn read_sequence(reader: &mut Reader, coll_type: CollType) -> Result<Value, ErrorWithSpan> {
    let open_span = reader.next()?.span.clone();
    let mut seq: Vec<Value> = vec![];

    loop {
        let token = match reader.peek() {
            Ok(t) => t.token.clone(),
            Err(_) => {
                return Err(ErrorWithSpan {
                    error: Error::SyntaxError(SyntaxError::UnbalancedDelimiter {
                        delimiter: match coll_type {
                            CollType::List => '(',
                            CollType::Vector => '[',
                            CollType::HashMap | CollType::Set => '{',
                        },
                        position: open_span.start,
                    }),
                    span: open_span.clone(),
                    source: reader.source.clone(),
                });
            }
        };
        match token {
            Token::RParen if coll_type == CollType::List => break,
            Token::RBracket if coll_type == CollType::Vector => break,
            Token::RBrace if matches!(coll_type, CollType::HashMap | CollType::Set) => break,
            _ => {
                seq.push(read_form(reader)?);
            }
        }
    }
    let _ = reader.next(); // consume closing delimiter
    match coll_type {
        CollType::List => Ok(Value::List(seq)),
        CollType::Vector => Ok(Value::Vector(seq)),
        CollType::HashMap => Ok(Value::Map(value::create_btree_map_from_sequence(seq))),
        CollType::Set => Ok(Value::Set(value::create_btree_set_from_sequence(seq))),
    }
}

/// Reads a single form from the reader and returns a Value.
fn read_form(reader: &mut Reader) -> Result<Value, ErrorWithSpan> {
    let token_ast = reader.peek()?;
    match token_ast.token {
        // --------- Quote ---------
        Token::Quote => {
            let _ = reader.next()?;
            Ok(list![Value::Symbol("quote".to_string()), read_form(reader)?])
        }

        // --------- ~ ---------
        Token::Tilde => {
            let _ = reader.next()?;
            Ok(list![Value::Symbol("unquote".to_string()), read_form(reader)?])
        }

        // --------- ~@ ---------
        Token::TildeAt => {
            let _ = reader.next()?;
            Ok(list![Value::Symbol("unquote-splicing".to_string()), read_form(reader)?])
        }

        // --------- # ---------
        Token::Hash => {
            let _ = reader.next()?; // remove the hash
            let next_token_ast = reader.peek()?;
            match next_token_ast.token {
                // Var-quote (#'sym)
                Token::Quote => {
                    let _ = reader.next()?;
                    Ok(list![Value::Symbol("var".to_string()), read_form(reader)?])
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
                    return Err(ErrorWithSpan {
                        error: Error::SyntaxError(SyntaxError::UnexpectedToken {
                            found: format!("{:?}", next_token_ast.token),
                            expected: "set".to_string(),
                        }),
                        span: next_token_ast.span.clone(),
                        source: reader.source.clone(),
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

pub fn read(source: &str) -> Result<Value, ErrorWithSpan> {
    let mut reader = tokenize(source);
    read_form(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_core_cases() {
        let src: &str = r#"(def x 123 -45.6 "hi\n\"there\"" ; comment
                      ~@ [a b] {k v} 'sym `q ^m @n #t _ ~ sym-1 \a)"#;

        let toks: Vec<TokenAST> = tokenize(src).tokens;

        // Smoke-check presence/order of a few key tokens:
        assert!(
            toks.starts_with(&[TokenAST { token: Token::LParen, span: Span { start: 0, end: 1 } }])
        );
        assert!(
            toks.contains(&TokenAST { token: Token::Int(123), span: Span { start: 1, end: 4 } })
        );
        assert!(
            toks.contains(&TokenAST {
                token: Token::Float(-45.6),
                span: Span { start: 5, end: 10 }
            })
        );
        assert!(toks.contains(&TokenAST {
            token: Token::Str("hi\n\"there\"".to_string()),
            span: Span { start: 11, end: 26 }
        }));
        assert!(
            toks.contains(&TokenAST { token: Token::TildeAt, span: Span { start: 27, end: 29 } })
        );
        assert!(
            toks.contains(&TokenAST { token: Token::LBracket, span: Span { start: 30, end: 32 } })
        );
        assert!(
            toks.contains(&TokenAST { token: Token::LBrace, span: Span { start: 33, end: 35 } })
        );
        assert!(
            toks.contains(&TokenAST { token: Token::Quote, span: Span { start: 36, end: 38 } })
        );
        assert!(
            toks.contains(&TokenAST { token: Token::Backtick, span: Span { start: 39, end: 41 } })
        );
        assert!(
            toks.contains(&TokenAST { token: Token::Caret, span: Span { start: 42, end: 44 } })
        );
        assert!(toks.contains(&TokenAST { token: Token::At, span: Span { start: 45, end: 47 } }));
        assert!(toks.contains(&TokenAST { token: Token::Hash, span: Span { start: 48, end: 50 } }));
        assert!(
            toks.contains(&TokenAST {
                token: Token::Underscore,
                span: Span { start: 51, end: 53 }
            })
        );
        assert!(
            toks.contains(&TokenAST { token: Token::Tilde, span: Span { start: 54, end: 56 } })
        );
        assert!(toks.contains(&TokenAST {
            token: Token::Symbol("sym-1".to_string()),
            span: Span { start: 57, end: 62 }
        }));
        assert!(toks.contains(&TokenAST {
            token: Token::Symbol("sym".to_string()),
            span: Span { start: 63, end: 66 }
        }));
        assert!(toks.contains(&TokenAST {
            token: Token::Symbol("\\a".to_string()),
            span: Span { start: 67, end: 69 }
        }));
        assert!(
            toks.contains(&TokenAST { token: Token::RParen, span: Span { start: 70, end: 71 } })
        );
    }

    #[test]
    fn unterminated_string_is_flagged() {
        let src = r#""oops"#;
        let v: Vec<_> = tokenize(src).tokens;
        assert_eq!(
            v,
            vec![TokenAST {
                token: Token::UnterminatedStr("\"oops".to_string()),
                span: Span { start: 0, end: 5 }
            }]
        );
    }
}
