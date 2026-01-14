//! Interactive REPL for Evolve.
//!
//! Features:
//! - Colored output for values and errors
//! - Command history with persistence
//! - Special commands: :help, :clear, :load, :quit
//! - Multi-line input detection

use colored::Colorize;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::path::Path;

use crate::reader::Source;
use crate::runtime::{Runtime, RuntimeRef};
use crate::{devtools, error::Diagnostic};

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct REPL {
    pub print_ast: bool,
    pub runtime: RuntimeRef,
}

impl REPL {
    pub fn new(print_ast: bool) -> Self {
        REPL { print_ast, runtime: Runtime::new() }
    }

    pub fn rep(&self, input: &str) -> Result<String, Diagnostic> {
        let value = Runtime::rep(self.runtime.clone(), input, Source::REPL)?;

        if self.print_ast {
            println!("{}", devtools::pretty_print_ast(&value));
        }

        Ok(value.to_string())
    }

    /// Handles special REPL commands that start with `:`.
    fn handle_command(&self, command: &str) -> Option<CommandResult> {
        let parts: Vec<&str> = command.trim().split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        match parts[0] {
            ":help" | ":h" | ":?" => Some(CommandResult::Print(self.help_message())),
            ":clear" | ":cls" => Some(CommandResult::Clear),
            ":quit" | ":exit" | ":q" => Some(CommandResult::Quit),
            ":load" | ":l" => {
                if parts.len() < 2 {
                    Some(CommandResult::Error(
                        "Usage: :load <file>".to_string(),
                    ))
                } else {
                    Some(self.load_file(parts[1]))
                }
            }
            ":ns" => {
                let ns = self.runtime.get_current_namespace();
                Some(CommandResult::Print(format!(
                    "Current namespace: {}",
                    ns.name.cyan()
                )))
            }
            ":vars" => Some(self.list_vars()),
            _ => None,
        }
    }

    fn help_message(&self) -> String {
        format!(
            r#"{}

{}
  :help, :h, :?     Show this help message
  :clear, :cls      Clear the screen
  :quit, :exit, :q  Exit the REPL
  :load <file>      Load and evaluate a file
  :ns               Show the current namespace
  :vars             List all defined vars in current namespace

{}
  Ctrl+C            Cancel current input
  Ctrl+D            Exit the REPL
  Up/Down arrows    Navigate history

{}
  (+ 1 2 3)         => 6
  (def x 42)        => var (bound):user/x
  (fn [x] (* x x))  => #<fn:anon>
"#,
            format!("Evolve REPL v{}", VERSION).bold(),
            "Commands:".yellow().bold(),
            "Keyboard Shortcuts:".yellow().bold(),
            "Examples:".yellow().bold()
        )
    }

    fn load_file(&self, path: &str) -> CommandResult {
        let file_path = Path::new(path);
        match fs::read_to_string(file_path) {
            Ok(source) => {
                match Runtime::rep(
                    self.runtime.clone(),
                    &source,
                    Source::File(file_path.to_path_buf()),
                ) {
                    Ok(value) => CommandResult::Print(format!(
                        "{} {}\n=> {}",
                        "Loaded:".green(),
                        path,
                        value
                    )),
                    Err(e) => CommandResult::Error(e.format()),
                }
            }
            Err(e) => {
                CommandResult::Error(format!("Failed to read file '{}': {}", path, e))
            }
        }
    }

    fn list_vars(&self) -> CommandResult {
        let ns = self.runtime.get_current_namespace();
        let mut output = format!("{} {}:\n", "Vars in".yellow(), ns.name.cyan());

        let bindings: Vec<_> = ns.all_bindings().collect();
        if bindings.is_empty() {
            output.push_str("  (no vars defined)");
        } else {
            for (sym, var) in bindings {
                let sym_name = crate::interner::sym_to_str(*sym);
                let bound = if var.is_bound() { "bound" } else { "unbound" };
                let visibility = if var.is_public() { "public" } else { "private" };
                output.push_str(&format!(
                    "  {} ({}, {})\n",
                    sym_name.green(),
                    bound,
                    visibility
                ));
            }
        }

        CommandResult::Print(output)
    }

    /// Checks if input appears to be incomplete (unclosed parens, brackets, etc.).
    fn is_incomplete(&self, input: &str) -> bool {
        let mut paren_depth = 0i32;
        let mut bracket_depth = 0i32;
        let mut brace_depth = 0i32;
        let mut in_string = false;
        let mut escape_next = false;

        for c in input.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '(' if !in_string => paren_depth += 1,
                ')' if !in_string => paren_depth -= 1,
                '[' if !in_string => bracket_depth += 1,
                ']' if !in_string => bracket_depth -= 1,
                '{' if !in_string => brace_depth += 1,
                '}' if !in_string => brace_depth -= 1,
                _ => {}
            }
        }

        paren_depth > 0 || bracket_depth > 0 || brace_depth > 0 || in_string
    }

    fn print_welcome(&self) {
        println!(
            "{}",
            format!(
                r#"
 ___________      .__
 \_   _____/__  __|  |___  __ ____
  |    __)_\  \/ /|  |\  \/ // __ \
  |        \\   / |  |_\   /\  ___/
 /_______  / \_/  |____/\_/  \___  >
         \/                      \/
                             v{}
"#,
                VERSION
            )
            .cyan()
        );
        println!(
            "Type {} for help, {} to exit.\n",
            ":help".yellow(),
            ":quit".yellow()
        );
    }

    pub fn run(&self) {
        let mut rl = DefaultEditor::new().unwrap();
        if rl.load_history(".evolve-history").is_err() {}

        self.print_welcome();

        let mut input_buffer = String::new();

        'repl_loop: loop {
            let current_namespace = self.runtime.get_current_namespace();
            let prompt = if input_buffer.is_empty() {
                format!("{}> ", current_namespace.name.green())
            } else {
                format!("{}. ", "..".dimmed())
            };

            let readline = rl.readline(&prompt);
            match readline {
                Ok(line) => {
                    // Check for commands (only when buffer is empty)
                    if input_buffer.is_empty() && line.trim().starts_with(':') {
                        if let Err(err) = rl.add_history_entry(line.as_str()) {
                            eprintln!("{}: {:?}", "History error".red(), err);
                        }

                        if let Some(result) = self.handle_command(&line) {
                            match result {
                                CommandResult::Print(msg) => println!("{}", msg),
                                CommandResult::Error(msg) => {
                                    eprintln!("{}: {}", "Error".red().bold(), msg);
                                }
                                CommandResult::Clear => {
                                    print!("\x1B[2J\x1B[1;1H");
                                    self.print_welcome();
                                }
                                CommandResult::Quit => break 'repl_loop,
                            }
                        }
                        continue 'repl_loop;
                    }

                    // Accumulate input
                    if !input_buffer.is_empty() {
                        input_buffer.push('\n');
                    }
                    input_buffer.push_str(&line);

                    // Check if input is complete
                    if self.is_incomplete(&input_buffer) {
                        continue 'repl_loop;
                    }

                    // Save to history
                    if let Err(err) = rl.add_history_entry(input_buffer.as_str()) {
                        eprintln!("{}: {:?}", "History error".red(), err);
                    }

                    if let Err(err) = rl.save_history(".evolve-history") {
                        eprintln!("{}: {:?}", "Save history error".red(), err);
                    }

                    // Evaluate the complete input
                    if !input_buffer.trim().is_empty() {
                        match self.rep(&input_buffer) {
                            Ok(out) => {
                                // Color the output based on type
                                let colored_out = self.colorize_output(&out);
                                println!("{}", colored_out);
                            }
                            Err(e) => {
                                println!("{}", e.format());
                            }
                        }
                    }

                    input_buffer.clear();
                }
                Err(ReadlineError::Interrupted) => {
                    if !input_buffer.is_empty() {
                        println!("{}", "Input cancelled".dimmed());
                        input_buffer.clear();
                    }
                    continue 'repl_loop;
                }
                Err(ReadlineError::Eof) => break 'repl_loop,
                Err(err) => {
                    println!("{}: {:?}", "Error".red(), err);
                    break 'repl_loop;
                }
            }
        }

        println!("\n{}", "Goodbye!".cyan());
    }

    /// Colorizes REPL output based on value type.
    fn colorize_output(&self, output: &str) -> String {
        // Simple heuristics to colorize output
        if output == "nil" {
            return output.dimmed().to_string();
        }
        if output == "true" {
            return output.green().to_string();
        }
        if output == "false" {
            return output.red().to_string();
        }
        if output.starts_with("#<") {
            return output.magenta().to_string();
        }
        if output.starts_with("var ") {
            return output.cyan().to_string();
        }
        if output.starts_with(':') {
            return output.yellow().to_string();
        }
        if output.starts_with('"') {
            return output.green().to_string();
        }
        if output.parse::<i64>().is_ok() || output.parse::<f64>().is_ok() {
            return output.blue().to_string();
        }

        output.to_string()
    }
}

enum CommandResult {
    Print(String),
    Error(String),
    Clear,
    Quit,
}
