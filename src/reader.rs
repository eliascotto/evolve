use logos::Logos;
use std::{fmt, ops::Range, path, sync::Arc};

use crate::collections::{List, Map, Set, Vector};
use crate::core::Symbol;
use crate::error::{Diagnostic, Error, SyntaxError};
use crate::interner::{self, SymId};
use crate::runtime::{Runtime, RuntimeRef};
use crate::utils;
use crate::value::{self, Value};

#[derive(Debug, PartialEq, Clone)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn from_range(span: Range<usize>) -> Self {
        Self::new(span.start, span.end)
    }

    pub fn to_range(&self) -> Range<usize> {
        self.start..self.end
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn shift(&self, offset: usize) -> Self {
        Self::new(self.start + offset, self.end + offset)
    }
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
    #[regex(r"\\.",
      priority = 2,
      callback = |lex| unescape_char(lex.slice()))]
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
    // [^\s\[\]{}('"`,;^@#~)]+
    #[regex(r#"[^ \t\r\n\[\]{}\('"`,;^@#~)]+"#,
      priority = 0,
      callback = |lex| lex.slice().to_owned())]
    Symbol(String),

    #[regex(r#":[^ \t\r\n\[\]{}\('"`,;^@#~)]+"#,
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
    file: Source,
}

#[derive(Debug, PartialEq, Clone)]
enum CollType {
    List,
    Vector,
    HashMap,
    Set,
}

#[derive(Debug)]
pub struct Reader {
    /// The tokens of the source code.
    tokens: Vec<TokenCST>,
    /// The source code string.
    source: String,
    /// The current position in the source code.
    position: usize,
    /// The file source of the source code, a file or the REPL.
    file: Source,
    /// A syntax quote reader instance.
    syntax_quote_reader: SyntaxQuoteReader,
}

impl Reader {
    /// Reads the source code transforming it into a Value.
    pub fn read(
        source: &str,
        file: Source,
        runtime: Arc<Runtime>,
    ) -> Result<Value, Diagnostic> {
        let mut reader = Self::tokenize(source, file, runtime);
        reader.read_form()
    }

    /// Transforms the source code into a vector of tokens,
    /// and returns a new Reader instance.
    fn tokenize(source_code: &str, file: Source, runtime: Arc<Runtime>) -> Self {
        let mut lexer = Token::lexer(source_code.trim());
        let mut tokens: Vec<TokenCST> = vec![];

        while let Some(token) = lexer.next() {
            if let Ok(token) = token {
                if let Token::Comment = token {
                    continue;
                }

                tokens.push(TokenCST {
                    token,
                    span: Span::from_range(lexer.span()),
                    file: file.clone(),
                });
            }
        }

        Reader {
            tokens,
            source: source_code.to_string(),
            position: 0,
            file: file,
            syntax_quote_reader: SyntaxQuoteReader::new(runtime),
        }
    }

    /// Reads the next token from the reader and returns a TokenAST. Increments the position.
    fn next(&mut self) -> Result<&TokenCST, Diagnostic> {
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
    fn peek(&self) -> Result<&TokenCST, Diagnostic> {
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
        self.tokens.last().map(|t| t.span.clone()).unwrap_or_else(|| {
            // If there are no tokens, return a span at the end of the source
            let end = self.source.len();
            Span { start: end, end }
        })
    }

    /// Reads an atom from the reader and returns a Value.
    /// An atom is a single value that is not a collection.
    /// It can be a symbol, a number, a string, a character, or a keyword.
    fn read_atom(&mut self) -> Result<Value, Diagnostic> {
        let token_ast = self.next()?;
        match &token_ast.token {
            Token::Symbol(symbol) => match symbol.as_str() {
                "nil" => Ok(Value::Nil { span: token_ast.span.clone() }),
                "true" => {
                    Ok(Value::Bool { span: token_ast.span.clone(), value: true })
                }
                "false" => {
                    Ok(Value::Bool { span: token_ast.span.clone(), value: false })
                }
                // Ordinary symbol
                _ => {
                    // Create a symbol to get its id and namespace
                    let symbol = Symbol::new(symbol.as_str());
                    // Then return the symbol as a Value
                    Ok(Value::Symbol {
                        span: token_ast.span.clone(),
                        value: Arc::new(symbol),
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

            Token::Float(float) => Ok(Value::Float {
                span: token_ast.span.clone(),
                value: float.clone(),
            }),

            Token::Str(str) => Ok(Value::String {
                span: token_ast.span.clone(),
                value: Arc::from(str.clone()),
            }),

            Token::Char(char) => {
                Ok(Value::Char { span: token_ast.span.clone(), value: char.clone() })
            }

            Token::UnterminatedStr(_) => Err(Diagnostic {
                error: Error::SyntaxError(SyntaxError::UnterminatedString),
                span: token_ast.span.clone(),
                source: self.source.clone(),
                file: self.file.clone(),
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
                source: self.source.clone(),
                file: self.file.clone(),
                secondary_spans: None,
                notes: None,
            }),
        }
    }

    /// Reads a sequence of tokens and returns a Value of the given collection type.
    fn read_sequence(&mut self, coll_type: CollType) -> Result<Value, Diagnostic> {
        let open_span = self.next()?.span.clone();
        let mut seq: Vec<Value> = vec![];

        loop {
            let token = match self.peek() {
                Ok(t) => t.token.clone(),
                Err(_) => {
                    return Err(Diagnostic {
                        error: Error::SyntaxError(
                            SyntaxError::UnbalancedDelimiter {
                                delimiter: match coll_type {
                                    CollType::List => '(',
                                    CollType::Vector => '[',
                                    CollType::HashMap | CollType::Set => '{',
                                },
                                position: open_span.start,
                            },
                        ),
                        span: self.last_span(), // Where EOF/unexpected token is
                        source: self.source.clone(),
                        file: self.file.clone(),
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
                    seq.push(self.read_form()?);
                }
            }
        }

        let close_span = self.next()?.span.clone(); // consume closing delimiter
        // Span should cover the entire sequence, including the opening and closing delimiters
        let seq_span = Span { start: open_span.start, end: close_span.end };
        match coll_type {
            CollType::List => Ok(Value::List {
                span: seq_span,
                value: Arc::new(List::from_iter(seq)),
                meta: None,
            }),
            CollType::Vector => Ok(Value::Vector {
                span: seq_span,
                value: Arc::new(Vector::from_iter(seq)),
                meta: None,
            }),
            CollType::HashMap => {
                let mut map = Map::new();
                let mut iter = seq.into_iter();
                while let Some(key) = iter.next() {
                    if let Some(value) = iter.next() {
                        map = map.insert(key, value);
                    } else {
                        return Err(Diagnostic {
                            error: Error::SyntaxError(SyntaxError::InvalidMap {
                                reason: "Invalid map: odd number of elements"
                                    .to_string(),
                            }),
                            span: seq_span,
                            source: self.source.clone(),
                            file: self.file.clone(),
                            secondary_spans: Some(vec![open_span.clone()]),
                            notes: None,
                        });
                    }
                }
                Ok(Value::Map { span: seq_span, value: Arc::new(map), meta: None })
            }
            CollType::Set => Ok(Value::Set {
                span: seq_span,
                value: Arc::new(Set::from_iter(seq)),
                meta: None,
            }),
        }
    }

    /// Splits a symbol that is immediately after `#` and begins with an underscore
    /// into an explicit underscore token followed by the remainder re-tokenized.
    fn split_hash_discard_symbol(
        &mut self,
        symbol_index: usize,
        symbol_token: TokenCST,
    ) {
        let Token::Symbol(symbol_str) = symbol_token.token else { return };
        if !symbol_str.starts_with('_') {
            return;
        }

        let rest_offset = symbol_token.span.start + 1;
        self.tokens[symbol_index] = TokenCST {
            token: Token::Underscore,
            span: Span { start: symbol_token.span.start, end: rest_offset },
            file: symbol_token.file.clone(),
        };

        let rest = &symbol_str[1..];
        if rest.is_empty() {
            return;
        }

        let mut rest_tokens: Vec<TokenCST> = Vec::new();
        let mut lexer = Token::lexer(rest);

        while let Some(token_result) = lexer.next() {
            if let Ok(token) = token_result {
                if let Token::Comment = token {
                    continue;
                }
                let rest_span = lexer.span();
                rest_tokens.push(TokenCST {
                    token,
                    span: Span {
                        start: rest_offset + rest_span.start,
                        end: rest_offset + rest_span.end,
                    },
                    file: symbol_token.file.clone(),
                });
            }
        }

        if !rest_tokens.is_empty() {
            self.tokens.splice(symbol_index + 1..symbol_index + 1, rest_tokens);
        }
    }

    /// Reads a single form from the reader and returns a Value.
    fn read_form(&mut self) -> Result<Value, Diagnostic> {
        let token_ast = self.peek()?;
        match token_ast.token {
            // --------- Quote ---------
            Token::Quote => {
                let quote_span = token_ast.span.clone();
                let _ = self.next()?;
                Ok(Value::List {
                    span: quote_span.clone(),
                    value: Arc::new(List::from_iter(vec![
                        value::symbol(
                            interner::intern_sym("quote"),
                            None,
                            None,
                            quote_span,
                        ),
                        self.read_form()?,
                    ])),
                    meta: None,
                })
            }

            // --------- ` ---------
            Token::Backtick => {
                let _ = self.next()?;
                let form = self.read_form()?;
                Ok(self.syntax_quote_reader.syntax_quote(&form))
            }

            // --------- ~ ---------
            Token::Tilde => {
                let tilde_span = token_ast.span.clone();
                let _ = self.next()?;
                Ok(Value::List {
                    span: tilde_span.clone(),
                    value: Arc::new(List::from_iter(vec![
                        value::symbol(
                            interner::intern_sym("unquote"),
                            None,
                            None,
                            tilde_span.clone(),
                        ),
                        self.read_form()?,
                    ])),
                    meta: None,
                })
            }

            // --------- ~@ ---------
            Token::TildeAt => {
                let tilde_at_span = token_ast.span.clone();
                let _ = self.next()?;
                Ok(Value::List {
                    span: tilde_at_span.clone(),
                    value: Arc::new(List::from_iter(vec![
                        value::symbol(
                            interner::intern_sym("unquote-splicing"),
                            None,
                            None,
                            tilde_at_span.clone(),
                        ),
                        self.read_form()?,
                    ])),
                    meta: None,
                })
            }

            // --------- ^ ---------
            Token::Caret => {
                let caret_span = token_ast.span.clone();
                let _ = self.next()?; // consume the caret
                let meta = self.read_form()?;
                let meta_span = match &meta {
                    Value::Keyword { span, .. } => span.clone(),
                    Value::Map { span, .. } => span.clone(),
                    _ => caret_span.clone(),
                };

                let next_object = self.read_form()?;
                let object_with_meta =
                    next_object.set_meta(meta).map_err(|e| Diagnostic {
                        error: e,
                        span: Span { start: caret_span.start, end: meta_span.end },
                        source: self.source.clone(),
                        file: self.file.clone(),
                        secondary_spans: None,
                        notes: None,
                    })?;
                Ok(object_with_meta)
            }

            // --------- # ---------
            Token::Hash => {
                let _ = self.next()?; // remove the hash
                loop {
                    let next_token_ast = self.peek()?.clone();
                    match &next_token_ast.token {
                        // Var-quote (#'sym)
                        Token::Quote => {
                            let var_quote_span = next_token_ast.span.clone();
                            let _ = self.next()?;
                            break Ok(Value::List {
                                span: var_quote_span.clone(),
                                value: Arc::new(List::from_iter(vec![
                                    value::symbol(
                                        interner::intern_sym("var"),
                                        None,
                                        None,
                                        var_quote_span,
                                    ),
                                    self.read_form()?,
                                ])),
                                meta: None,
                            });
                        }
                        // Ignore next form (#_ symbol)
                        Token::Underscore => {
                            let _ = self.next()?;
                            let _ = self.read_form()?; // discard next form
                            break self.read_form();
                        }
                        // Create a set (#{..})
                        Token::LBrace => break self.read_sequence(CollType::Set),
                        // Split hash discard symbol (#_sym)
                        Token::Symbol(symbol) if symbol.starts_with('_') => {
                            let idx = self.position;
                            // #_sym -> #_ sym
                            // then next iteration will discard sym using Token::Underscore rule
                            self.split_hash_discard_symbol(idx, next_token_ast);
                            continue;
                        }
                        _ => {
                            break Err(Diagnostic {
                                error: Error::SyntaxError(
                                    SyntaxError::UnexpectedToken {
                                        found: format!("{:?}", next_token_ast.token),
                                        expected: "set".to_string(),
                                    },
                                ),
                                span: next_token_ast.span.clone(),
                                source: self.source.clone(),
                                file: self.file.clone(),
                                secondary_spans: None,
                                notes: None,
                            });
                        }
                    }
                }
            }

            // --------- Sequences ---------
            Token::LParen => self.read_sequence(CollType::List),
            Token::LBracket => self.read_sequence(CollType::Vector),
            Token::LBrace => self.read_sequence(CollType::HashMap),
            _ => self.read_atom(),
        }
    }
}

//===----------------------------------------------------------------------===//
// SyntaxQuoteReader
//===----------------------------------------------------------------------===//

#[derive(Debug)]
struct SyntaxQuoteReader {
    gensyms: Map<SymId, SymId>,
    runtime: RuntimeRef,
}

impl SyntaxQuoteReader {
    fn new(runtime: RuntimeRef) -> Self {
        Self { gensyms: Map::new(), runtime }
    }

    pub fn syntax_quote(&mut self, form: &Value) -> Value {
        match form {
            Value::Symbol { span, value: sym } => {
                // Quote special forms (quote def)
                if self.runtime.evaluator.is_special_form(sym.id()) {
                    return self.make_quote_list(
                        value::symbol(
                            sym.id(),
                            sym.ns,
                            sym.metadata(),
                            span.clone(),
                        ),
                        span,
                    );
                }

                let gensym_value = self.handle_symbol(sym, span);
                self.make_quote_list(gensym_value, span)
            }
            Value::List { span, value, .. } => {
                if value.is_empty() {
                    return form.clone();
                }

                self.make_seq_concat_list(value, span)
            }
            _ => form.clone(),
        }
    }

    /// Creates a `(quote value)` list.
    fn make_quote_list(&self, value: Value, span: &Span) -> Value {
        value::list_from_vec(
            vec![
                self.quote_symbol(span.clone()),
                value,
            ],
            span.clone(),
        )
    }

    /// Creates a symbol for the `quote` special form.
    fn quote_symbol(&self, span: Span) -> Value {
        value::symbol(
            self.runtime.evaluator.special_forms.s_quote,
            None,
            None,
            span,
        )
    }

    /// Handles symbol resolution and gensym generation.
    fn handle_symbol(&mut self, sym: &Arc<Symbol>, span: &Span) -> Value {
        // Gensyms are unqualified symbols that end with "#"
        if !sym.is_qualified() && sym.name().ends_with("#") {
            self.get_or_create_gensym(sym, span)
        } else {
            // Resolve symbol
            let mut sym_clone = Arc::as_ref(sym).clone();
            let resolved = self.runtime.resolve_symbol(&mut sym_clone);
            Value::Symbol {
                span: span.clone(),
                value: Arc::new(resolved),
            }
        }
    }

    /// Gets an existing gensym or creates a new one.
    fn get_or_create_gensym(&mut self, sym: &Arc<Symbol>, span: &Span) -> Value {
        if let Some(gensym_id) = self.gensyms.get(&sym.id()) {
            // Gensym is already interned
            value::symbol(*gensym_id, None, None, span.clone())
        } else {
            // Intern a new symbol: symname__231__auto__
            let gensym_name = format!(
                "{}__{}__auto__",
                utils::remove_last_char(sym.name()),
                self.runtime.next_id()
            );
            let new_sym = Symbol::new(gensym_name.as_str());
            self.gensyms = self.gensyms.insert(sym.id(), new_sym.id());
            Value::Symbol {
                span: span.clone(),
                value: Arc::new(new_sym),
            }
        }
    }

    /// Creates a `(seq (concat ...))` list for syntax-quoted lists.
    fn make_seq_concat_list(&mut self, list: &Arc<List<Value>>, span: &Span) -> Value {
        let seq_sym = self.runtime.get_native_fn_sym("seq");
        let concat_sym = self.runtime.get_native_fn_sym("concat");

        let concat_args = {
            let mut args = vec![value::symbol(concat_sym, None, None, span.clone())];
            args.extend(self.expand_list(list));
            args
        };

        value::list_from_vec(
            vec![
                value::symbol(seq_sym, None, None, span.clone()),
                value::list_from_vec(concat_args, span.clone()),
            ],
            span.clone(),
        )
    }

    /// Builds the `(concat ...)` arguments that a syntax-quoted list expands to.
    /// Plain elements become `(list (syntax-quote elem))`, `~` emits `(list elem)`
    /// and `~@` injects the spliced form directly.
    fn expand_list(&mut self, list: &List<Value>) -> Vec<Value> {
        let list_sym = self.runtime.get_native_fn_sym("list");
        let unquote_sym = interner::intern_sym("unquote");
        let unquote_splicing_sym = interner::intern_sym("unquote-splicing");

        let wrap_in_list = |expr: Value, span: &Span| -> Value {
            value::list_from_vec(
                vec![
                    value::symbol(list_sym, None, None, span.clone()),
                    expr,
                ],
                span.clone(),
            )
        };

        list.iter()
            .map(|item| {
                let item_span = item.span();
                self.handle_list_item(item, &item_span, unquote_sym, unquote_splicing_sym, &wrap_in_list)
            })
            .collect()
    }

    /// Handles a single item in a syntax-quoted list, processing unquote/unquote-splicing.
    fn handle_list_item(
        &mut self,
        item: &Value,
        item_span: &Span,
        unquote_sym: SymId,
        unquote_splicing_sym: SymId,
        wrap_in_list: &dyn Fn(Value, &Span) -> Value,
    ) -> Value {
        if let Value::List { value: inner, .. } = item {
            if let Some((head, tail)) = inner.split_first() {
                if head.match_symbol(unquote_sym) {
                    // (~ x) => (list x)
                    let expr = tail.head().cloned().unwrap_or_else(|| Value::Nil {
                        span: item_span.clone(),
                    });
                    return wrap_in_list(expr, item_span);
                }
                if head.match_symbol(unquote_splicing_sym) {
                    // (~@ xs) => xs
                    return tail.head().cloned().unwrap_or_else(|| Value::Nil {
                        span: item_span.clone(),
                    });
                }
            }
        }

        // Default case: recursively syntax-quote the element and make it a single-element chunk.
        wrap_in_list(self.syntax_quote(item), item_span)
    }
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Unescapes a single character from an escape sequence.
/// Handles common escape sequences: \n, \t, \r, \", \\, and others.
/// For unknown escape sequences, returns the character after the backslash.
fn unescape_char(s: &str) -> char {
    // The slice should be "\X" where X is the character after the backslash
    if s.len() >= 2 && s.starts_with('\\') {
        match s.chars().nth(1) {
            Some('n') => '\n',
            Some('t') => '\t',
            Some('r') => '\r',
            Some('"') => '"',
            Some('\\') => '\\',
            Some('0') => '\0',
            Some(ch) => ch, // For unknown escape sequences, return the character itself
            None => '\\',   // Shouldn't happen, but handle it
        }
    } else {
        // Fallback: try to parse as-is (shouldn't happen with correct regex)
        s.chars().next().unwrap_or('?')
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Arc<Runtime> {
        Runtime::new()
    }

    fn assert_symbol(value: &Value, expected: SymId) {
        match value {
            Value::Symbol { value: sym, .. } => {
                assert_eq!(
                    sym.id(),
                    expected,
                    "expected symbol id {}, got {}",
                    interner::sym_to_str(expected),
                    interner::sym_to_str(sym.id())
                );
            }
            other => panic!("expected Symbol, got {:?}", other),
        }
    }

    fn as_list<'a>(value: &'a Value) -> &'a List<Value> {
        match value {
            Value::List { value: list, .. } => list.as_ref(),
            other => panic!("expected List, got {:?}", other),
        }
    }

    fn assert_quoted_symbol_chunk(
        chunk: &Value,
        list_sym: SymId,
        quote_sym: SymId,
        expected: SymId,
    ) {
        let chunk_list = as_list(chunk);
        let mut chunk_iter = chunk_list.iter();
        assert_symbol(chunk_iter.next().expect("list symbol"), list_sym);
        let quoted = chunk_iter.next().expect("quoted form");
        assert!(chunk_iter.next().is_none(), "chunk should have two elements");

        let quoted_list = as_list(quoted);
        let mut quoted_iter = quoted_list.iter();
        assert_symbol(quoted_iter.next().expect("quote symbol"), quote_sym);
        assert_symbol(quoted_iter.next().expect("quoted symbol"), expected);
        assert!(quoted_iter.next().is_none(), "quoted form should have two elements");
    }

    fn assert_unquote_chunk(chunk: &Value, list_sym: SymId, expected: SymId) {
        let chunk_list = as_list(chunk);
        let mut chunk_iter = chunk_list.iter();
        assert_symbol(chunk_iter.next().expect("list symbol"), list_sym);
        assert_symbol(chunk_iter.next().expect("unquoted symbol"), expected);
        assert!(chunk_iter.next().is_none(), "chunk should have two elements");
    }

    #[test]
    fn tokenizes_core_cases() {
        let src: &str = r#"(def x 123 -45.6 "hi\n\"there\"" ; comment
                      ~@ [a b] {k v} 'sym `q ^m @n #t _ ~ sym-1 \a)"#;

        let toks: Vec<TokenCST> =
            Reader::tokenize(src, Source::REPL, setup()).tokens;

        // Smoke-check presence/order of a few key tokens:
        assert!(toks.starts_with(&[TokenCST {
            token: Token::LParen,
            span: Span { start: 0, end: 1 },
            file: Source::REPL,
        }]));
        assert!(toks.contains(&TokenCST {
            token: Token::Int(123),
            span: Span { start: 7, end: 10 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Float(-45.6),
            span: Span { start: 11, end: 16 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Str("hi\n\"there\"".to_string()),
            span: Span { start: 17, end: 32 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::TildeAt,
            span: Span { start: 65, end: 67 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::LBracket,
            span: Span { start: 68, end: 69 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::LBrace,
            span: Span { start: 74, end: 75 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Quote,
            span: Span { start: 80, end: 81 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Backtick,
            span: Span { start: 85, end: 86 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Caret,
            span: Span { start: 88, end: 89 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::At,
            span: Span { start: 91, end: 92 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Hash,
            span: Span { start: 94, end: 95 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Underscore,
            span: Span { start: 97, end: 98 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Tilde,
            span: Span { start: 99, end: 100 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Symbol("sym-1".to_string()),
            span: Span { start: 101, end: 106 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::Char('a'),
            span: Span { start: 107, end: 109 },
            file: Source::REPL,
        }));
        assert!(toks.contains(&TokenCST {
            token: Token::RParen,
            span: Span { start: 109, end: 110 },
            file: Source::REPL,
        }));
    }

    #[test]
    fn unterminated_string_is_flagged() {
        let src = r#""oops"#;
        let v: Vec<_> = Reader::tokenize(src, Source::REPL, setup()).tokens;
        assert_eq!(
            v,
            vec![TokenCST {
                token: Token::UnterminatedStr("\"oops".to_string()),
                span: Span { start: 0, end: 5 },
                file: Source::REPL,
            }]
        );
    }

    //===----------------------------------------------------------------------===//
    // Reading Atoms
    //===----------------------------------------------------------------------===//

    #[test]
    fn reads_symbols() {
        let result = Reader::read("my-symbol", Source::REPL, setup()).unwrap();
        match result {
            Value::Symbol { value, .. } => {
                assert_eq!(interner::sym_to_str(value.id()), "my-symbol");
            }
            _ => panic!("Expected Symbol, got {:?}", result),
        }
    }

    #[test]
    fn reads_keywords() {
        let result = Reader::read(":keyword", Source::REPL, setup()).unwrap();
        match result {
            Value::Keyword { value, .. } => {
                assert_eq!(interner::kw_print(value), ":keyword");
            }
            _ => panic!("Expected Keyword, got {:?}", result),
        }
    }

    #[test]
    fn reads_integers() {
        let result = Reader::read("42", Source::REPL, setup()).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int, got {:?}", result),
        }

        let result = Reader::read("-123", Source::REPL, setup()).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, -123),
            _ => panic!("Expected Int, got {:?}", result),
        }
    }

    #[test]
    fn reads_floats() {
        let result = Reader::read("3.14", Source::REPL, setup()).unwrap();
        match result {
            Value::Float { value, .. } => {
                assert!((value - 3.14).abs() < f64::EPSILON)
            }
            _ => panic!("Expected Float, got {:?}", result),
        }

        let result = Reader::read("-0.5", Source::REPL, setup()).unwrap();
        match result {
            Value::Float { value, .. } => {
                assert!((value - -0.5).abs() < f64::EPSILON)
            }
            _ => panic!("Expected Float, got {:?}", result),
        }
    }

    #[test]
    fn reads_strings() {
        let result = Reader::read(r#""hello""#, Source::REPL, setup()).unwrap();
        match result {
            Value::String { value, .. } => assert_eq!(value, Arc::from("hello")),
            _ => panic!("Expected String, got {:?}", result),
        }
    }

    #[test]
    fn reads_strings_with_escapes() {
        let result =
            Reader::read(r#""hello\nworld""#, Source::REPL, setup()).unwrap();
        match result {
            Value::String { value, .. } => {
                assert_eq!(value, Arc::from("hello\nworld"))
            }
            _ => panic!("Expected String, got {:?}", result),
        }

        let result = Reader::read(r#""tab\there""#, Source::REPL, setup()).unwrap();
        match result {
            Value::String { value, .. } => assert_eq!(value, Arc::from("tab\there")),
            _ => panic!("Expected String, got {:?}", result),
        }

        let result =
            Reader::read(r#""quote\"here""#, Source::REPL, setup()).unwrap();
        match result {
            Value::String { value, .. } => {
                assert_eq!(value, Arc::from("quote\"here"))
            }
            _ => panic!("Expected String, got {:?}", result),
        }

        let result =
            Reader::read(r#""backslash\\here""#, Source::REPL, setup()).unwrap();
        match result {
            Value::String { value, .. } => {
                assert_eq!(value, Arc::from("backslash\\here"))
            }
            _ => panic!("Expected String, got {:?}", result),
        }
    }

    #[test]
    fn reads_characters() {
        let result = Reader::read(r"\a", Source::REPL, setup()).unwrap();
        match result {
            Value::Char { value, .. } => assert_eq!(value, 'a'),
            _ => panic!("Expected Char, got {:?}", result),
        }

        let result = Reader::read(r"\n", Source::REPL, setup()).unwrap();
        match result {
            Value::Char { value, .. } => assert_eq!(value, '\n'),
            _ => panic!("Expected Char, got {:?}", result),
        }

        let result = Reader::read(r"\t", Source::REPL, setup()).unwrap();
        match result {
            Value::Char { value, .. } => assert_eq!(value, '\t'),
            _ => panic!("Expected Char, got {:?}", result),
        }
    }

    #[test]
    fn reads_booleans() {
        let result = Reader::read("true", Source::REPL, setup()).unwrap();
        match result {
            Value::Bool { value, .. } => assert_eq!(value, true),
            _ => panic!("Expected Bool(true), got {:?}", result),
        }

        let result = Reader::read("false", Source::REPL, setup()).unwrap();
        match result {
            Value::Bool { value, .. } => assert_eq!(value, false),
            _ => panic!("Expected Bool(false), got {:?}", result),
        }
    }

    #[test]
    fn reads_nil() {
        let result = Reader::read("nil", Source::REPL, setup()).unwrap();
        match result {
            Value::Nil { .. } => {}
            _ => panic!("Expected Nil, got {:?}", result),
        }
    }

    #[test]
    fn unterminated_string_returns_error() {
        let result = Reader::read(r#""unterminated"#, Source::REPL, setup());
        assert!(result.is_err());
        match result {
            Err(diagnostic) => match diagnostic.error {
                Error::SyntaxError(SyntaxError::UnterminatedString) => {}
                _ => panic!(
                    "Expected UnterminatedString error, got {:?}",
                    diagnostic.error
                ),
            },
            _ => panic!("Expected error, got Ok"),
        }
    }

    //===----------------------------------------------------------------------===//
    // Reading Sequences
    //===----------------------------------------------------------------------===//

    #[test]
    fn reads_empty_list() {
        let result = Reader::read("()", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => assert_eq!(value.len(), 0),
            _ => panic!("Expected empty List, got {:?}", result),
        }
    }

    #[test]
    fn reads_list_with_elements() {
        let result = Reader::read("(1 2 3)", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => {
                assert_eq!(value.len(), 3);
                let items: Vec<_> = value.iter().collect();
                match items[0] {
                    Value::Int { value: v, .. } => assert_eq!(*v, 1),
                    _ => panic!("Expected Int(1)"),
                }
                match items[1] {
                    Value::Int { value: v, .. } => assert_eq!(*v, 2),
                    _ => panic!("Expected Int(2)"),
                }
                match items[2] {
                    Value::Int { value: v, .. } => assert_eq!(*v, 3),
                    _ => panic!("Expected Int(3)"),
                }
            }
            _ => panic!("Expected List, got {:?}", result),
        }
    }

    #[test]
    fn reads_nested_lists() {
        let result = Reader::read("(1 (2 3) 4)", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => {
                assert_eq!(value.len(), 3);
                let items: Vec<_> = value.iter().collect();
                match items[1] {
                    Value::List { value: inner, .. } => {
                        assert_eq!(inner.len(), 2);
                        let inner_items: Vec<_> = inner.iter().collect();
                        match inner_items[0] {
                            Value::Int { value, .. } => assert_eq!(*value, 2),
                            _ => panic!("Expected Int(2)"),
                        }
                        match inner_items[1] {
                            Value::Int { value, .. } => assert_eq!(*value, 3),
                            _ => panic!("Expected Int(3)"),
                        }
                    }
                    _ => panic!("Expected nested List"),
                }
            }
            _ => panic!("Expected List, got {:?}", result),
        }
    }

    #[test]
    fn reads_empty_vector() {
        let result = Reader::read("[]", Source::REPL, setup()).unwrap();
        match result {
            Value::Vector { value, .. } => assert_eq!(value.len(), 0),
            _ => panic!("Expected empty Vector, got {:?}", result),
        }
    }

    #[test]
    fn reads_vector_with_elements() {
        let result = Reader::read("[a b c]", Source::REPL, setup()).unwrap();
        match result {
            Value::Vector { value, .. } => {
                assert_eq!(value.len(), 3);
                match value.get(0) {
                    Some(Value::Symbol { value: v, .. }) => {
                        assert_eq!(interner::sym_to_str(v.id()), "a");
                    }
                    _ => panic!("Expected Symbol(a)"),
                }
            }
            _ => panic!("Expected Vector, got {:?}", result),
        }
    }

    #[test]
    fn reads_empty_map() {
        let result = Reader::read("{}", Source::REPL, setup()).unwrap();
        match result {
            Value::Map { value, .. } => assert_eq!(value.len(), 0),
            _ => panic!("Expected empty Map, got {:?}", result),
        }
    }

    #[test]
    fn reads_map_with_key_value_pairs() {
        let result = Reader::read("{:a 1 :b 2}", Source::REPL, setup()).unwrap();
        match result {
            Value::Map { value, .. } => {
                assert_eq!(value.len(), 2);
                // Check that keys exist
                let key_a = Value::Keyword {
                    span: Span { start: 0, end: 0 },
                    value: interner::intern_kw(":a"),
                };
                let key_b = Value::Keyword {
                    span: Span { start: 0, end: 0 },
                    value: interner::intern_kw(":b"),
                };
                assert!(value.contains_key(&key_a));
                assert!(value.contains_key(&key_b));
            }
            _ => panic!("Expected Map, got {:?}", result),
        }
    }

    #[test]
    fn reads_empty_set() {
        let result = Reader::read("#{}", Source::REPL, setup()).unwrap();
        match result {
            Value::Set { value, .. } => assert_eq!(value.len(), 0),
            _ => panic!("Expected empty Set, got {:?}", result),
        }
    }

    #[test]
    fn reads_set_with_elements() {
        let result = Reader::read("#{1 2 3}", Source::REPL, setup()).unwrap();
        match result {
            Value::Set { value, .. } => {
                assert_eq!(value.len(), 3);
                let int1 = Value::Int { span: Span { start: 0, end: 0 }, value: 1 };
                let int2 = Value::Int { span: Span { start: 0, end: 0 }, value: 2 };
                let int3 = Value::Int { span: Span { start: 0, end: 0 }, value: 3 };
                assert!(value.contains(&int1));
                assert!(value.contains(&int2));
                assert!(value.contains(&int3));
            }
            _ => panic!("Expected Set, got {:?}", result),
        }
    }

    //===----------------------------------------------------------------------===//
    // Reader Macros
    //===----------------------------------------------------------------------===//

    #[test]
    fn reads_quote() {
        let result = Reader::read("'symbol", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => {
                assert_eq!(value.len(), 2);
                let items: Vec<_> = value.iter().collect();
                match items[0] {
                    Value::Symbol { value: v, .. } => {
                        assert_eq!(interner::sym_to_str(v.id()), "quote");
                    }
                    _ => panic!("Expected quote symbol"),
                }
                match items[1] {
                    Value::Symbol { .. } => {}
                    _ => panic!("Expected quoted symbol"),
                }
            }
            _ => panic!("Expected List (quote form), got {:?}", result),
        }
    }

    #[test]
    fn reads_tilde() {
        let result = Reader::read("~symbol", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => {
                assert_eq!(value.len(), 2);
                let items: Vec<_> = value.iter().collect();
                match items[0] {
                    Value::Symbol { value: v, .. } => {
                        assert_eq!(interner::sym_to_str(v.id()), "unquote");
                    }
                    _ => panic!("Expected unquote symbol"),
                }
            }
            _ => panic!("Expected List (unquote form), got {:?}", result),
        }
    }

    #[test]
    fn reads_tilde_at() {
        let result = Reader::read("~@symbol", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => {
                assert_eq!(value.len(), 2);
                let items: Vec<_> = value.iter().collect();
                match items[0] {
                    Value::Symbol { value: v, .. } => {
                        assert_eq!(interner::sym_to_str(v.id()), "unquote-splicing");
                    }
                    _ => panic!("Expected unquote-splicing symbol"),
                }
            }
            _ => panic!("Expected List (unquote-splicing form), got {:?}", result),
        }
    }

    #[test]
    fn reads_var_quote() {
        let result = Reader::read("#'symbol", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => {
                assert_eq!(value.len(), 2);
                let items: Vec<_> = value.iter().collect();
                match items[0] {
                    Value::Symbol { value: v, .. } => {
                        assert_eq!(interner::sym_to_str(v.id()), "var");
                    }
                    _ => panic!("Expected var symbol"),
                }
            }
            _ => panic!("Expected List (var form), got {:?}", result),
        }
    }

    #[test]
    fn reads_hash_underscore_discard() {
        // #_ should discard the next form and return the one after
        let result =
            Reader::read("#_discard-me keep-me", Source::REPL, setup()).unwrap();
        match result {
            Value::Symbol { value, .. } => {
                assert_eq!(interner::sym_to_str(value.id()), "keep-me");
            }
            _ => panic!("Expected Symbol(keep-me), got {:?}", result),
        }

        let vector_with_space =
            Reader::read("[1 #_ 2 3]", Source::REPL, setup()).unwrap();
        match vector_with_space {
            Value::Vector { value, .. } => {
                assert_eq!(value.len(), 2);
                match value.get(0) {
                    Some(Value::Int { value: int, .. }) => assert_eq!(*int, 1),
                    other => panic!("Expected Int(1), got {:?}", other),
                }
                match value.get(1) {
                    Some(Value::Int { value: int, .. }) => assert_eq!(*int, 3),
                    other => panic!("Expected Int(3), got {:?}", other),
                }
            }
            other => panic!("Expected Vector, got {:?}", other),
        }

        let vector_without_space =
            Reader::read("[1 #_2 3]", Source::REPL, setup()).unwrap();
        match vector_without_space {
            Value::Vector { value, .. } => {
                assert_eq!(value.len(), 2);
                match value.get(0) {
                    Some(Value::Int { value: int, .. }) => assert_eq!(*int, 1),
                    other => panic!("Expected Int(1), got {:?}", other),
                }
                match value.get(1) {
                    Some(Value::Int { value: int, .. }) => assert_eq!(*int, 3),
                    other => panic!("Expected Int(3), got {:?}", other),
                }
            }
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    //===----------------------------------------------------------------------===//
    // Error Cases
    //===----------------------------------------------------------------------===//

    // TODO: `last_span()` panics when tokens is empty. Need to handle empty token list
    // in error reporting before this test can pass.
    #[test]
    fn unbalanced_list_returns_error() {
        let result = Reader::read("(1 2 3", Source::REPL, setup());
        assert!(result.is_err());
        match result {
            Err(diagnostic) => match diagnostic.error {
                Error::SyntaxError(SyntaxError::UnbalancedDelimiter {
                    delimiter,
                    ..
                }) => {
                    assert_eq!(delimiter, '(');
                }
                _ => panic!(
                    "Expected UnbalancedDelimiter error, got {:?}",
                    diagnostic.error
                ),
            },
            _ => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn unbalanced_vector_returns_error() {
        let result = Reader::read("[1 2 3", Source::REPL, setup());
        assert!(result.is_err());
        match result {
            Err(diagnostic) => match diagnostic.error {
                Error::SyntaxError(SyntaxError::UnbalancedDelimiter {
                    delimiter,
                    ..
                }) => {
                    assert_eq!(delimiter, '[');
                }
                _ => panic!(
                    "Expected UnbalancedDelimiter error, got {:?}",
                    diagnostic.error
                ),
            },
            _ => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn unbalanced_map_returns_error() {
        let result = Reader::read("{:a 1", Source::REPL, setup());
        assert!(result.is_err());
        match result {
            Err(diagnostic) => match diagnostic.error {
                Error::SyntaxError(SyntaxError::UnbalancedDelimiter {
                    delimiter,
                    ..
                }) => {
                    assert_eq!(delimiter, '{');
                }
                _ => panic!(
                    "Expected UnbalancedDelimiter error, got {:?}",
                    diagnostic.error
                ),
            },
            _ => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn map_with_odd_number_of_forms_returns_error() {
        // Test with 1 element (odd)
        let result = Reader::read("{:a}", Source::REPL, setup());
        assert!(result.is_err());
        match result {
            Err(diagnostic) => match diagnostic.error {
                Error::SyntaxError(SyntaxError::InvalidMap { reason }) => {
                    assert_eq!(reason, "Invalid map: odd number of elements");
                }
                _ => panic!("Expected InvalidMap error, got {:?}", diagnostic.error),
            },
            _ => panic!("Expected error, got Ok"),
        }

        // Test with 3 elements (odd)
        let result = Reader::read("{:a 1 :b}", Source::REPL, setup());
        assert!(result.is_err());
        match result {
            Err(diagnostic) => match diagnostic.error {
                Error::SyntaxError(SyntaxError::InvalidMap { reason }) => {
                    assert_eq!(reason, "Invalid map: odd number of elements");
                }
                _ => panic!("Expected InvalidMap error, got {:?}", diagnostic.error),
            },
            _ => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn unexpected_eof_returns_error() {
        let result = Reader::read("", Source::REPL, setup());
        assert!(result.is_err());
        match result {
            Err(diagnostic) => match diagnostic.error {
                Error::SyntaxError(SyntaxError::UnexpectedEOF { .. }) => {}
                _ => panic!(
                    "Expected UnexpectedEOF error, got {:?}",
                    diagnostic.error
                ),
            },
            _ => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn unexpected_eof_in_quote_returns_error() {
        let result = Reader::read("'", Source::REPL, setup());
        assert!(result.is_err());
        match result {
            Err(diagnostic) => match diagnostic.error {
                Error::SyntaxError(SyntaxError::UnexpectedEOF { .. }) => {}
                _ => panic!(
                    "Expected UnexpectedEOF error, got {:?}",
                    diagnostic.error
                ),
            },
            _ => panic!("Expected error, got Ok"),
        }
    }

    //===----------------------------------------------------------------------===//
    // Complex Cases
    //===----------------------------------------------------------------------===//

    #[test]
    fn reads_complex_nested_structure() {
        let result =
            Reader::read("(defn add [x y] (+ x y))", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => {
                assert_eq!(value.len(), 4);
                let items: Vec<_> = value.iter().collect();
                // Check first element is 'defn'
                match items[0] {
                    Value::Symbol { value: v, .. } => {
                        assert_eq!(interner::sym_to_str(v.id()), "defn");
                    }
                    _ => panic!("Expected Symbol(defn)"),
                }
                // Check third element is a vector
                match items[2] {
                    Value::Vector { value: params, .. } => {
                        assert_eq!(params.len(), 2);
                    }
                    _ => panic!("Expected Vector of parameters"),
                }
            }
            _ => panic!("Expected List, got {:?}", result),
        }
    }

    #[test]
    fn reads_quoted_list() {
        let result = Reader::read("'(1 2 3)", Source::REPL, setup()).unwrap();
        match result {
            Value::List { value, .. } => {
                assert_eq!(value.len(), 2);
                let items: Vec<_> = value.iter().collect();
                match items[0] {
                    Value::Symbol { value: v, .. } => {
                        assert_eq!(interner::sym_to_str(v.id()), "quote");
                    }
                    _ => panic!("Expected quote symbol"),
                }
                match items[1] {
                    Value::List { value: inner, .. } => {
                        assert_eq!(inner.len(), 3);
                    }
                    _ => panic!("Expected quoted list"),
                }
            }
            _ => panic!("Expected List (quote form), got {:?}", result),
        }
    }

    #[test]
    fn reads_map_with_nested_structures() {
        let result = Reader::read(
            "{:nested {:a 1 :b 2} :list [1 2 3]}",
            Source::REPL,
            setup(),
        )
        .unwrap();
        match result {
            Value::Map { value, .. } => {
                assert_eq!(value.len(), 2);
                // Verify structure is valid
            }
            _ => panic!("Expected Map, got {:?}", result),
        }
    }

    #[test]
    fn reads_multiple_forms_with_whitespace() {
        // Note: Reader::read only reads one form, so this should read just the first
        let result = Reader::read("  42  ", Source::REPL, setup()).unwrap();
        match result {
            Value::Int { value, .. } => assert_eq!(value, 42),
            _ => panic!("Expected Int(42), got {:?}", result),
        }
    }

    #[test]
    fn reads_symbols_with_special_chars() {
        let result =
            Reader::read("symbol-with-dashes", Source::REPL, setup()).unwrap();
        match result {
            Value::Symbol { value, .. } => {
                assert_eq!(interner::sym_to_str(value.id()), "symbol-with-dashes");
            }
            _ => panic!("Expected Symbol, got {:?}", result),
        }

        let result =
            Reader::read("symbol_with_underscores", Source::REPL, setup()).unwrap();
        match result {
            Value::Symbol { value, .. } => {
                assert_eq!(
                    interner::sym_to_str(value.id()),
                    "symbol_with_underscores"
                );
            }
            _ => panic!("Expected Symbol, got {:?}", result),
        }
    }

    #[test]
    fn reads_keywords_with_namespace() {
        let result = Reader::read(":ns/keyword", Source::REPL, setup()).unwrap();
        match result {
            Value::Keyword { value, .. } => {
                assert_eq!(interner::kw_print(value), ":ns/keyword");
            }
            _ => panic!("Expected Keyword, got {:?}", result),
        }
    }

    #[test]
    fn syntax_quote_list_wraps_seq_and_concat() {
        let runtime = setup();
        let result = Reader::read("`(a b)", Source::REPL, runtime.clone()).unwrap();

        let seq_sym = runtime.get_native_fn_sym("seq");
        let concat_sym = runtime.get_native_fn_sym("concat");
        let list_sym = runtime.get_native_fn_sym("list");
        let quote_sym = runtime.evaluator.special_forms.s_quote;
        let sym_a = interner::intern_sym("a");
        let sym_b = interner::intern_sym("b");

        let outer = as_list(&result);
        let mut outer_iter = outer.iter();
        assert_symbol(outer_iter.next().expect("seq symbol"), seq_sym);
        let concat_form = outer_iter.next().expect("concat form");
        assert!(outer_iter.next().is_none(), "outer form should have two elements");

        let concat_list = as_list(concat_form);
        let mut concat_iter = concat_list.iter();
        assert_symbol(concat_iter.next().expect("concat symbol"), concat_sym);
        let chunks: Vec<_> = concat_iter.collect();
        assert_eq!(chunks.len(), 2, "expected two chunks for `(a b)`");

        let mut chunk_iter = chunks.into_iter();
        assert_quoted_symbol_chunk(
            chunk_iter.next().expect("first chunk"),
            list_sym,
            quote_sym,
            sym_a,
        );
        assert_quoted_symbol_chunk(
            chunk_iter.next().expect("second chunk"),
            list_sym,
            quote_sym,
            sym_b,
        );
        assert!(chunk_iter.next().is_none(), "unexpected extra chunk");
    }

    #[test]
    fn syntax_quote_handles_unquote() {
        let runtime = setup();
        let result = Reader::read("`(a ~b c)", Source::REPL, runtime.clone()).unwrap();

        let seq_sym = runtime.get_native_fn_sym("seq");
        let concat_sym = runtime.get_native_fn_sym("concat");
        let list_sym = runtime.get_native_fn_sym("list");
        let quote_sym = runtime.evaluator.special_forms.s_quote;
        let sym_a = interner::intern_sym("a");
        let sym_b = interner::intern_sym("b");
        let sym_c = interner::intern_sym("c");

        let outer = as_list(&result);
        let mut outer_iter = outer.iter();
        assert_symbol(outer_iter.next().expect("seq symbol"), seq_sym);
        let concat_form = outer_iter.next().expect("concat form");
        assert!(outer_iter.next().is_none(), "outer form should have two elements");

        let concat_list = as_list(concat_form);
        let mut concat_iter = concat_list.iter();
        assert_symbol(concat_iter.next().expect("concat symbol"), concat_sym);
        let chunks: Vec<_> = concat_iter.collect();
        assert_eq!(chunks.len(), 3, "expected three chunks for `(a ~b c)`");

        let mut chunk_iter = chunks.into_iter();
        assert_quoted_symbol_chunk(
            chunk_iter.next().expect("first chunk"),
            list_sym,
            quote_sym,
            sym_a,
        );
        assert_unquote_chunk(
            chunk_iter.next().expect("unquote chunk"),
            list_sym,
            sym_b,
        );
        assert_quoted_symbol_chunk(
            chunk_iter.next().expect("last chunk"),
            list_sym,
            quote_sym,
            sym_c,
        );
        assert!(chunk_iter.next().is_none(), "unexpected extra chunk");
    }

    #[test]
    fn syntax_quote_handles_unquote_splicing() {
        let runtime = setup();
        let result =
            Reader::read("`(a ~@b c)", Source::REPL, runtime.clone()).unwrap();

        let seq_sym = runtime.get_native_fn_sym("seq");
        let concat_sym = runtime.get_native_fn_sym("concat");
        let list_sym = runtime.get_native_fn_sym("list");
        let quote_sym = runtime.evaluator.special_forms.s_quote;
        let sym_a = interner::intern_sym("a");
        let sym_b = interner::intern_sym("b");
        let sym_c = interner::intern_sym("c");

        let outer = as_list(&result);
        let mut outer_iter = outer.iter();
        assert_symbol(outer_iter.next().expect("seq symbol"), seq_sym);
        let concat_form = outer_iter.next().expect("concat form");
        assert!(outer_iter.next().is_none(), "outer form should have two elements");

        let concat_list = as_list(concat_form);
        let mut concat_iter = concat_list.iter();
        assert_symbol(concat_iter.next().expect("concat symbol"), concat_sym);
        let chunks: Vec<_> = concat_iter.collect();
        assert_eq!(chunks.len(), 3, "expected three chunks for `(a ~@b c)`");

        let mut chunk_iter = chunks.into_iter();
        assert_quoted_symbol_chunk(
            chunk_iter.next().expect("first chunk"),
            list_sym,
            quote_sym,
            sym_a,
        );
        assert_symbol(chunk_iter.next().expect("spliced chunk"), sym_b);
        assert_quoted_symbol_chunk(
            chunk_iter.next().expect("last chunk"),
            list_sym,
            quote_sym,
            sym_c,
        );
        assert!(chunk_iter.next().is_none(), "unexpected extra chunk");
    }
}
