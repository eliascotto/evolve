use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::{error::ErrorWithSpan, reader};

pub struct REPL {}

impl REPL {
    pub fn rep(&self, input: &str) -> Result<String, ErrorWithSpan> {
        let ast = reader::read(input)?;
        Ok(ast.to_string())
    }

    pub fn run(&self) {
        let mut rl = DefaultEditor::new().unwrap();
        if rl.load_history(".evolve-history").is_err() {}

        'repl_loop: loop {
            let readline = rl.readline(&format!("> "));
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
                                println!("{}", e.format_error());
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
