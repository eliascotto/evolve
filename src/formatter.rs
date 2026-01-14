//! Code formatter for Evolve.
//!
//! Formats Evolve source code with consistent style:
//! - 2-space indentation
//! - Line breaks for long forms
//! - Consistent spacing

use crate::error::Diagnostic;
use crate::reader::{Reader, Source};
use crate::runtime::RuntimeRef;
use crate::value::Value;
use crate::interner;

/// Maximum line length before wrapping
const MAX_LINE_LENGTH: usize = 80;

/// Indentation width
const INDENT_WIDTH: usize = 2;

/// Code formatter.
pub struct Formatter {
    runtime: RuntimeRef,
}

impl Formatter {
    pub fn new(runtime: RuntimeRef) -> Self {
        Self { runtime }
    }

    /// Formats source code and returns the formatted result.
    pub fn format(&self, source: &str, file: Source) -> Result<String, Diagnostic> {
        // Parse the source code to get AST
        let ast = Reader::read(source, file, self.runtime.clone())?;

        // Format the AST back to a string
        Ok(self.format_value(&ast, 0))
    }

    /// Formats a single value with the given indentation level.
    fn format_value(&self, value: &Value, indent: usize) -> String {
        match value {
            Value::Nil { .. } => "nil".to_string(),
            Value::Bool { value: b, .. } => b.to_string(),
            Value::Int { value: n, .. } => n.to_string(),
            Value::Float { value: f, .. } => {
                let s = f.to_string();
                // Ensure floats have a decimal point
                if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    format!("{}.0", s)
                } else {
                    s
                }
            }
            Value::Char { value: c, .. } => format!("\\{}", c),
            Value::String { value: s, .. } => format!("\"{}\"", escape_string(s)),
            Value::Symbol { value: sym, .. } => {
                if sym.is_qualified() {
                    format!("{}/{}", sym.namespace(), sym.name())
                } else {
                    sym.name().to_string()
                }
            }
            Value::Keyword { value: kw, .. } => interner::kw_print(*kw),
            Value::List { value: list, meta, .. } => {
                self.format_list(list.iter().collect(), "(", ")", indent, meta.is_some())
            }
            Value::Vector { value: vec, meta, .. } => {
                self.format_list(vec.iter().collect(), "[", "]", indent, meta.is_some())
            }
            Value::Map { value: map, meta, .. } => {
                let pairs: Vec<&Value> = map.iter().flat_map(|(k, v)| [k, v]).collect();
                self.format_list(pairs, "{", "}", indent, meta.is_some())
            }
            Value::Set { value: set, meta, .. } => {
                self.format_list(set.iter().collect(), "#{", "}", indent, meta.is_some())
            }
            _ => value.to_string(),
        }
    }

    /// Formats a list-like structure (list, vector, map, set).
    fn format_list(
        &self,
        items: Vec<&Value>,
        open: &str,
        close: &str,
        indent: usize,
        _has_meta: bool,
    ) -> String {
        if items.is_empty() {
            return format!("{}{}", open, close);
        }

        // First, try to format on a single line
        let single_line = self.format_single_line(&items, open, close);
        if single_line.len() <= MAX_LINE_LENGTH && !single_line.contains('\n') {
            return single_line;
        }

        // Check for special forms that have specific formatting rules
        if open == "(" {
            if let Some(formatted) = self.format_special_form(&items, indent) {
                return formatted;
            }
        }

        // Multi-line formatting
        self.format_multi_line(&items, open, close, indent)
    }

    /// Formats items on a single line.
    fn format_single_line(&self, items: &[&Value], open: &str, close: &str) -> String {
        let formatted: Vec<String> =
            items.iter().map(|v| self.format_value(v, 0)).collect();
        format!("{}{}{}", open, formatted.join(" "), close)
    }

    /// Formats items on multiple lines.
    fn format_multi_line(
        &self,
        items: &[&Value],
        open: &str,
        close: &str,
        indent: usize,
    ) -> String {
        let inner_indent = indent + INDENT_WIDTH;
        let indent_str = " ".repeat(inner_indent);

        let mut result = String::new();
        result.push_str(open);

        for (i, item) in items.iter().enumerate() {
            if i == 0 {
                result.push_str(&self.format_value(item, inner_indent));
            } else {
                result.push('\n');
                result.push_str(&indent_str);
                result.push_str(&self.format_value(item, inner_indent));
            }
        }

        result.push_str(close);
        result
    }

    /// Handles special form formatting (def, defn, fn, let, if, etc.).
    fn format_special_form(&self, items: &[&Value], indent: usize) -> Option<String> {
        if items.is_empty() {
            return None;
        }

        let first = items[0];
        let sym_name = match first {
            Value::Symbol { value: sym, .. } => sym.name().to_string(),
            _ => return None,
        };

        match sym_name.as_str() {
            "def" | "defn" | "defmacro" => self.format_def(items, indent),
            "fn" | "fn*" => self.format_fn(items, indent),
            "let" | "let*" | "loop" => self.format_let(items, indent),
            "if" => self.format_if(items, indent),
            "cond" => self.format_cond(items, indent),
            "do" => self.format_do(items, indent),
            "ns" => self.format_ns(items, indent),
            _ => None,
        }
    }

    /// Formats def/defn/defmacro forms.
    fn format_def(&self, items: &[&Value], indent: usize) -> Option<String> {
        if items.len() < 2 {
            return None;
        }

        let inner_indent = indent + INDENT_WIDTH;
        let indent_str = " ".repeat(inner_indent);

        let keyword = self.format_value(items[0], indent);
        let name = self.format_value(items[1], indent);

        let mut result = format!("({} {}", keyword, name);

        // Check if this is a simple def or defn with params
        if items.len() > 2 {
            for item in &items[2..] {
                let formatted = self.format_value(item, inner_indent);
                if formatted.len() + result.lines().last().map(|l| l.len()).unwrap_or(0) > MAX_LINE_LENGTH {
                    result.push('\n');
                    result.push_str(&indent_str);
                } else {
                    result.push(' ');
                }
                result.push_str(&formatted);
            }
        }

        result.push(')');
        Some(result)
    }

    /// Formats fn/fn* forms.
    fn format_fn(&self, items: &[&Value], indent: usize) -> Option<String> {
        if items.len() < 2 {
            return None;
        }

        let inner_indent = indent + INDENT_WIDTH;
        let indent_str = " ".repeat(inner_indent);

        let keyword = self.format_value(items[0], indent);

        // Check if fn has a name (fn name [params] body)
        let (params_idx, name_str) = match items.get(1) {
            Some(Value::Symbol { .. }) => (2, Some(self.format_value(items[1], indent))),
            Some(Value::Vector { .. }) => (1, None),
            _ => return None,
        };

        let params = items.get(params_idx).map(|p| self.format_value(p, indent));

        let mut result = format!("({}", keyword);
        if let Some(name) = name_str {
            result.push(' ');
            result.push_str(&name);
        }
        if let Some(p) = params {
            result.push(' ');
            result.push_str(&p);
        }

        // Body
        for item in items.iter().skip(params_idx + 1) {
            let formatted = self.format_value(item, inner_indent);
            result.push('\n');
            result.push_str(&indent_str);
            result.push_str(&formatted);
        }

        result.push(')');
        Some(result)
    }

    /// Formats let/let*/loop forms.
    fn format_let(&self, items: &[&Value], indent: usize) -> Option<String> {
        if items.len() < 2 {
            return None;
        }

        let inner_indent = indent + INDENT_WIDTH;
        let indent_str = " ".repeat(inner_indent);

        let keyword = self.format_value(items[0], indent);
        let bindings = self.format_bindings(items[1], inner_indent);

        let mut result = format!("({} {}", keyword, bindings);

        // Body
        for item in items.iter().skip(2) {
            let formatted = self.format_value(item, inner_indent);
            result.push('\n');
            result.push_str(&indent_str);
            result.push_str(&formatted);
        }

        result.push(')');
        Some(result)
    }

    /// Formats binding vectors with proper alignment.
    fn format_bindings(&self, value: &Value, indent: usize) -> String {
        match value {
            Value::Vector { value: vec, .. } => {
                let items: Vec<&Value> = vec.iter().collect();
                if items.is_empty() {
                    return "[]".to_string();
                }

                // Try single line first
                let single = self.format_single_line(&items, "[", "]");
                if single.len() <= MAX_LINE_LENGTH {
                    return single;
                }

                // Format binding pairs on separate lines
                let inner_indent = indent + 1;
                let indent_str = " ".repeat(inner_indent);

                let mut result = String::from("[");
                let mut i = 0;
                while i < items.len() {
                    if i > 0 {
                        result.push('\n');
                        result.push_str(&indent_str);
                    }
                    result.push_str(&self.format_value(items[i], inner_indent));
                    if i + 1 < items.len() {
                        result.push(' ');
                        result.push_str(&self.format_value(items[i + 1], inner_indent));
                    }
                    i += 2;
                }
                result.push(']');
                result
            }
            _ => self.format_value(value, indent),
        }
    }

    /// Formats if forms.
    fn format_if(&self, items: &[&Value], indent: usize) -> Option<String> {
        if items.len() < 3 {
            return None;
        }

        let inner_indent = indent + INDENT_WIDTH;
        let indent_str = " ".repeat(inner_indent);

        let keyword = self.format_value(items[0], indent);
        let condition = self.format_value(items[1], inner_indent);
        let then_branch = self.format_value(items[2], inner_indent);

        let mut result = format!("({} {}", keyword, condition);

        // Check if fits on one line
        let total_len = result.len() + then_branch.len() + items.get(3).map(|_| 20).unwrap_or(0);
        if total_len <= MAX_LINE_LENGTH && !then_branch.contains('\n') {
            result.push(' ');
            result.push_str(&then_branch);
            if let Some(else_branch) = items.get(3) {
                result.push(' ');
                result.push_str(&self.format_value(else_branch, inner_indent));
            }
        } else {
            result.push('\n');
            result.push_str(&indent_str);
            result.push_str(&then_branch);
            if let Some(else_branch) = items.get(3) {
                result.push('\n');
                result.push_str(&indent_str);
                result.push_str(&self.format_value(else_branch, inner_indent));
            }
        }

        result.push(')');
        Some(result)
    }

    /// Formats cond forms.
    fn format_cond(&self, items: &[&Value], indent: usize) -> Option<String> {
        if items.len() < 2 {
            return None;
        }

        let inner_indent = indent + INDENT_WIDTH;
        let indent_str = " ".repeat(inner_indent);

        let keyword = self.format_value(items[0], indent);

        let mut result = format!("({}", keyword);

        // Each clause on its own line
        for item in items.iter().skip(1) {
            result.push('\n');
            result.push_str(&indent_str);
            result.push_str(&self.format_value(item, inner_indent));
        }

        result.push(')');
        Some(result)
    }

    /// Formats do forms.
    fn format_do(&self, items: &[&Value], indent: usize) -> Option<String> {
        if items.len() < 2 {
            return None;
        }

        let inner_indent = indent + INDENT_WIDTH;
        let indent_str = " ".repeat(inner_indent);

        let keyword = self.format_value(items[0], indent);

        let mut result = format!("({}", keyword);

        for item in items.iter().skip(1) {
            result.push('\n');
            result.push_str(&indent_str);
            result.push_str(&self.format_value(item, inner_indent));
        }

        result.push(')');
        Some(result)
    }

    /// Formats ns forms.
    fn format_ns(&self, items: &[&Value], indent: usize) -> Option<String> {
        if items.len() < 2 {
            return None;
        }

        let inner_indent = indent + INDENT_WIDTH;
        let indent_str = " ".repeat(inner_indent);

        let keyword = self.format_value(items[0], indent);
        let name = self.format_value(items[1], indent);

        let mut result = format!("({} {}", keyword, name);

        // Each require/import clause on its own line
        for item in items.iter().skip(2) {
            result.push('\n');
            result.push_str(&indent_str);
            result.push_str(&self.format_value(item, inner_indent));
        }

        result.push(')');
        Some(result)
    }
}

/// Escapes special characters in a string.
fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            _ => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    fn format(source: &str) -> String {
        let rt = Runtime::new();
        let formatter = Formatter::new(rt);
        formatter.format(source, Source::REPL).unwrap()
    }

    #[test]
    fn test_format_primitives() {
        assert_eq!(format("nil"), "nil");
        assert_eq!(format("true"), "true");
        assert_eq!(format("false"), "false");
        assert_eq!(format("42"), "42");
        assert_eq!(format("3.14"), "3.14");
    }

    #[test]
    fn test_format_strings() {
        assert_eq!(format("\"hello\""), "\"hello\"");
        assert_eq!(format("\"hello\\nworld\""), "\"hello\\nworld\"");
    }

    #[test]
    fn test_format_keywords() {
        assert_eq!(format(":foo"), ":foo");
        assert_eq!(format(":foo/bar"), ":foo/bar");
    }

    #[test]
    fn test_format_simple_list() {
        assert_eq!(format("(+ 1 2)"), "(+ 1 2)");
        assert_eq!(format("(foo bar baz)"), "(foo bar baz)");
    }

    #[test]
    fn test_format_vector() {
        assert_eq!(format("[1 2 3]"), "[1 2 3]");
        assert_eq!(format("[]"), "[]");
    }

    #[test]
    fn test_format_map() {
        // Note: map ordering may vary
        let formatted = format("{:a 1}");
        assert!(formatted.contains(":a") && formatted.contains("1"));
    }

    #[test]
    fn test_format_def() {
        assert_eq!(format("(def x 42)"), "(def x 42)");
    }

    #[test]
    fn test_format_is_idempotent() {
        let source = "(def my-fn (fn [x y] (+ x y)))";
        let first = format(source);
        let second = format(&first);
        assert_eq!(first, second);
    }
}
