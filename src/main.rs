extern crate itertools;
extern crate logos;
extern crate rustc_hash;
extern crate rustyline;

use evolve::devtools;
use evolve::error::{Diagnostic, Error};
use evolve::reader::Source;
use evolve::runtime::Runtime;
use evolve::repl::REPL;
use std::{env, fs, path::Path};

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
enum ArgCmd {
    REPL { print_ast: bool },
    File { path: String, print_ast: bool },
    Help,
}

fn print_usage() {
    println!("Evolve v{}\n\n", VERSION);
    println!("Usage:");
    println!("  evolve                    Start the REPL");
    println!("  evolve --file <path>      Execute a file");
    println!(
        "  evolve --print-ast        Pretty-print AST before output (works with REPL and --file)"
    );
    println!("  evolve -h                 Show this help message");
}

fn parse_args(args: Vec<String>) -> Result<ArgCmd, String> {
    if args.len() == 1 {
        return Ok(ArgCmd::REPL { print_ast: false });
    }

    let mut print_ast = false;
    let mut file_path: Option<String> = None;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                return Ok(ArgCmd::Help);
            }
            "--print-ast" => {
                print_ast = true;
            }
            "--file" => {
                if i + 1 >= args.len() {
                    return Err("Error: --file requires a file path".to_string());
                }
                file_path = Some(args[i + 1].clone());
                i += 1; // Skip the file path
            }
            arg => {
                return Err(format!("Error: Unknown argument '{}'", arg));
            }
        }
        i += 1;
    }

    if let Some(path) = file_path {
        Ok(ArgCmd::File { path, print_ast })
    } else {
        Ok(ArgCmd::REPL { print_ast })
    }
}

fn run_file(file_path: &str, print_ast: bool) -> Result<(), Diagnostic> {
    let path = Path::new(file_path);
    let source_code = fs::read_to_string(path).map_err(|e| Diagnostic {
        error: Error::RuntimeError(format!("Failed to read file: {}", e)),
        span: 0..0,
        source: String::new(),
        file: Source::File(path.to_path_buf()),
        secondary_spans: None,
        notes: None,
    })?;

    let rt = Runtime::new();
    let ast = rt.rep(&source_code, Source::File(path.to_path_buf()))?;

    if print_ast {
        println!("{}", devtools::pretty_print_ast(&ast));
    }

    println!("{}", ast.to_string());
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let command = match parse_args(args) {
        Ok(cmd) => cmd,
        Err(e) => {
            eprintln!("{}\n\n", e);
            print_usage();
            std::process::exit(1);
        }
    };

    match command {
        ArgCmd::Help => {
            print_usage();
        }
        ArgCmd::REPL { print_ast } => {
            let repl = REPL::new(print_ast);
            repl.run();
        }
        ArgCmd::File { path, print_ast } => match run_file(&path, print_ast) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("{}", e.format());
                std::process::exit(1);
            }
        },
    }
}
