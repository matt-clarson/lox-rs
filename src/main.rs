use std::path::Path;

use clap::Parser;

fn main() {
    let cli = Cli::parse();

    let config = lox::Config { debug: cli.debug };

    if cli.debug {
        println!("!!!Starting in debug mode!!!\n");
    }

    let result = cli
        .file
        .map(|file| lox::interpret(file, &config))
        .unwrap_or_else(|| lox::Repl::run(&config));

    if let Err(err) = result {
        eprintln!("Error: {err}");
    }
}

#[derive(Parser)]
#[command(author, version, long_about = None)]
#[command(about = "Lox interpreter")]
struct Cli {
    /// Source file to read and execute - do not provide a file to instead launch the REPL.
    file: Option<Box<Path>>,

    /// Run in debug mode.
    #[arg(short, long)]
    debug: bool,
}
