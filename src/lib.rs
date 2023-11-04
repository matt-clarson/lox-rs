//! Implementation of the Lox programming language in Rust, based on the guide in [Crafting
//! Interpreters](https://craftinginterpreters.com/) by Robert Nystrom.
//!
//! For general usage, the [compiler](crate::compiler::Compiler) will handle instantiating scanning
//! and parsing code.
//!
//! Quick start:
//!
//! ```no_run
//! use lox;
//!
//! let config = lox::Config::default();
//!
//! lox::interpret("file.lox", &config).unwrap();
//! ```
//!
//! To manage the compiler and vm directly:
//!
//! ```
//! use lox::{compiler::Compiler, vm::{Value, VirtualMachine}};
//!
//! let compiler = Compiler::default();
//! let mut vm = VirtualMachine::default();
//!
//! let source = "2 * (5 - 2);";
//! let ops = compiler.compile(source).unwrap();
//!
//! // need to make sure the compiler output is loaded into the vm before calling `next`.
//! vm.load(ops);
//!
//! assert_eq!(vm.next(), Some(Ok(Value::Number(6.0))));
//! ```

#[cfg(test)]
#[macro_use]
extern crate assert_matches;

pub mod ast;
pub mod compiler;
pub mod parser;
pub mod repl;
pub mod scanner;
pub mod vm;

use std::{error::Error, fs, path::Path};

use compiler::Compiler;
pub use repl::Repl;
use vm::VirtualMachine;

/// Shared config for the Lox compiler and vm.
#[derive(Default)]
pub struct Config {
    /// Enable debugging.
    pub debug: bool,
}

/// Reads, compiles, and executes a single file of Lox source code.
pub fn interpret<P: AsRef<Path>>(file: P, config: &Config) -> Result<(), Box<dyn Error>> {
    let compiler = Compiler::default();
    let mut vm = VirtualMachine::default();
    if config.debug {
        vm.enable_debug();
    }

    let source = fs::read_to_string(file)?;

    let ops = compiler.compile(&source)?;
    vm.load(ops);

    for result in vm {
        match result {
            Ok(value) => println!("{value}"),
            Err(err) => return Err(err.into()),
        };
    }

    Ok(())
}
