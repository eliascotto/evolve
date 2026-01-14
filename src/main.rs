extern crate itertools;
extern crate logos;
extern crate rustc_hash;
extern crate rustyline;

use clap::{Parser, Subcommand};
use std::{fs, path::Path, process};

use evolve::devtools;
use evolve::error::{Diagnostic, Error};
use evolve::formatter::Formatter;
use evolve::reader::{Source, Span};
use evolve::repl::REPL;
use evolve::runtime::Runtime;
use evolve::test_runner::TestRunner;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(name = "evolve")]
#[command(author = "Elia Scotto <hello@scotto.me>")]
#[command(version = VERSION)]
#[command(about = "Evolve - A modern Lisp for the Rust ecosystem", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Pretty-print AST before output
    #[arg(long, global = true)]
    print_ast: bool,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the interactive REPL
    Repl,

    /// Execute an Evolve source file
    Run {
        /// Path to the source file
        file: String,
    },

    /// Run tests with :test metadata
    Test {
        /// Path to a file to test (optional, tests all if not specified)
        file: Option<String>,
    },

    /// Format Evolve source code
    #[command(alias = "fmt")]
    Format {
        /// Path to the source file to format
        file: String,

        /// Check formatting without modifying the file
        #[arg(long)]
        check: bool,
    },

    /// Compile to native code (requires codegen feature)
    Build {
        /// Path to the source file
        file: String,

        /// Output file path (defaults to input name with .o extension)
        #[arg(short, long)]
        output: Option<String>,

        /// Optimization level: debug, default, or release
        #[arg(long, default_value = "default")]
        opt: String,
    },
}

fn run_file(file_path: &str, print_ast: bool) -> Result<(), Diagnostic> {
    let path = Path::new(file_path);
    let source_code = fs::read_to_string(path).map_err(|e| Diagnostic {
        error: Error::RuntimeError(format!("Failed to read file: {}", e)),
        span: Span::new(0, 0),
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

fn run_tests(file_path: Option<&str>, verbose: bool) -> Result<(), Diagnostic> {
    let rt = Runtime::new();

    // If a file is specified, load it first
    if let Some(path) = file_path {
        let source_code = fs::read_to_string(path).map_err(|e| Diagnostic {
            error: Error::RuntimeError(format!("Failed to read file: {}", e)),
            span: Span::new(0, 0),
            source: String::new(),
            file: Source::File(Path::new(path).to_path_buf()),
            secondary_spans: None,
            notes: None,
        })?;

        rt.clone().rep(&source_code, Source::File(Path::new(path).to_path_buf()))?;
    }

    let runner = TestRunner::new(rt, verbose);
    let (passed, failed) = runner.run_all();

    if failed > 0 {
        Err(Diagnostic {
            error: Error::RuntimeError(format!(
                "{} test(s) failed, {} passed",
                failed, passed
            )),
            span: Span::new(0, 0),
            source: String::new(),
            file: Source::REPL,
            secondary_spans: None,
            notes: None,
        })
    } else {
        Ok(())
    }
}

fn format_file(file_path: &str, check_only: bool) -> Result<(), Diagnostic> {
    let path = Path::new(file_path);
    let source_code = fs::read_to_string(path).map_err(|e| Diagnostic {
        error: Error::RuntimeError(format!("Failed to read file: {}", e)),
        span: Span::new(0, 0),
        source: String::new(),
        file: Source::File(path.to_path_buf()),
        secondary_spans: None,
        notes: None,
    })?;

    let rt = Runtime::new();
    let formatter = Formatter::new(rt);
    let formatted = formatter.format(&source_code, Source::File(path.to_path_buf()))?;

    if check_only {
        if formatted == source_code {
            println!("File is properly formatted: {}", file_path);
            Ok(())
        } else {
            Err(Diagnostic {
                error: Error::RuntimeError(format!(
                    "File is not properly formatted: {}",
                    file_path
                )),
                span: Span::new(0, 0),
                source: String::new(),
                file: Source::File(path.to_path_buf()),
                secondary_spans: None,
                notes: None,
            })
        }
    } else {
        fs::write(path, &formatted).map_err(|e| Diagnostic {
            error: Error::RuntimeError(format!("Failed to write file: {}", e)),
            span: Span::new(0, 0),
            source: String::new(),
            file: Source::File(path.to_path_buf()),
            secondary_spans: None,
            notes: None,
        })?;
        println!("Formatted: {}", file_path);
        Ok(())
    }
}

#[cfg(feature = "codegen")]
fn build_file(
    file_path: &str,
    output: Option<&str>,
    opt_level: &str,
    print_ast: bool,
) -> Result<(), Diagnostic> {
    use evolve::codegen::{CodeGen, OptimizationConfig};
    use evolve::hir::HIR;
    use evolve::reader::Reader;

    let path = Path::new(file_path);
    let source_code = fs::read_to_string(path).map_err(|e| Diagnostic {
        error: Error::RuntimeError(format!("Failed to read file: {}", e)),
        span: Span::new(0, 0),
        source: String::new(),
        file: Source::File(path.to_path_buf()),
        secondary_spans: None,
        notes: None,
    })?;

    let rt = Runtime::new();

    // Parse the source code
    let ast = Reader::read(&source_code, Source::File(path.to_path_buf()), rt.clone())
        .map_err(|e| e)?;

    if print_ast {
        println!("{}", devtools::pretty_print_ast(&ast));
    }

    // Convert to HIR
    let hir = HIR::from_value(&ast).map_err(|e| Diagnostic {
        error: e.error,
        span: e.span,
        source: source_code.clone(),
        file: Source::File(path.to_path_buf()),
        secondary_spans: None,
        notes: None,
    })?;

    // Determine output path
    let output_path = match output {
        Some(p) => p.to_string(),
        None => {
            let stem = path.file_stem().unwrap().to_str().unwrap();
            format!("{}.o", stem)
        }
    };

    // Select optimization config
    let opt_config = match opt_level {
        "debug" => OptimizationConfig::debug(),
        "release" => OptimizationConfig::release(),
        _ => OptimizationConfig::default(),
    };

    // Generate code
    let codegen = CodeGen::new("evolve_module", opt_config);
    codegen.compile(&hir).map_err(|e| Diagnostic {
        error: e.error,
        span: e.span,
        source: source_code.clone(),
        file: Source::File(path.to_path_buf()),
        secondary_spans: None,
        notes: None,
    })?;

    // Write object file
    codegen.write_object(Path::new(&output_path)).map_err(|e| Diagnostic {
        error: Error::RuntimeError(format!("Failed to write object file: {}", e)),
        span: Span::new(0, 0),
        source: String::new(),
        file: Source::File(path.to_path_buf()),
        secondary_spans: None,
        notes: None,
    })?;

    println!("Compiled: {} -> {}", file_path, output_path);
    Ok(())
}

#[cfg(not(feature = "codegen"))]
fn build_file(
    _file_path: &str,
    _output: Option<&str>,
    _opt_level: &str,
    _print_ast: bool,
) -> Result<(), Diagnostic> {
    Err(Diagnostic {
        error: Error::RuntimeError(
            "The 'build' command requires the 'codegen' feature.\n\
             Recompile with: cargo build --features codegen"
                .to_string(),
        ),
        span: Span::new(0, 0),
        source: String::new(),
        file: Source::REPL,
        secondary_spans: None,
        notes: None,
    })
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Some(Commands::Repl) | None => {
            let repl = REPL::new(cli.print_ast);
            repl.run();
            Ok(())
        }
        Some(Commands::Run { file }) => run_file(&file, cli.print_ast),
        Some(Commands::Test { file }) => run_tests(file.as_deref(), cli.verbose),
        Some(Commands::Format { file, check }) => format_file(&file, check),
        Some(Commands::Build { file, output, opt }) => {
            build_file(&file, output.as_deref(), &opt, cli.print_ast)
        }
    };

    if let Err(e) = result {
        eprintln!("{}", e.format());
        process::exit(1);
    }
}
