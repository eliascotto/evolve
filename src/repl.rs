use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::reader::Source;
use crate::runtime::{Runtime, RuntimeRef};
use crate::{devtools, error::Diagnostic};

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

    pub fn run(&self) {
        let mut rl = DefaultEditor::new().unwrap();
        if rl.load_history(".evolve-history").is_err() {}

        'repl_loop: loop {
            let current_namespace = self.runtime.get_current_namespace();
            let repl_prompt = format!("{}> ", current_namespace.name);
            let readline = rl.readline(&repl_prompt);
            match readline {
                Ok(line) => {
                    if let Err(err) = rl.add_history_entry(line.as_str()) {
                        eprintln!("Error adding to history: {:?}", err);
                    }

                    if let Err(err) = rl.save_history(".evolve-history") {
                        eprintln!("Error saving history: {:?}", err);
                    }

                    if line.len() > 0 {
                        match self.rep(&line) {
                            Ok(out) => println!("{}", out),
                            Err(e) => {
                                println!("{}", e.format());
                                continue 'repl_loop;
                            }
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => continue 'repl_loop,
                Err(ReadlineError::Eof) => break 'repl_loop,
                Err(err) => {
                    println!("Error: {:?}", err);
                    break 'repl_loop;
                }
            }
        }
    }
}
