//! Implementation of the Lox programming language in Rust, based on the guide in [Crafting
//! Interpreters](https://craftinginterpreters.com/) by Robert Nystrom.
//!
//! For general usage, the [compiler](crate::compiler::Compiler) will handle instantiating scanning
//! and parsing code. See the example below:
//!
//! ```
//! use lox::{compiler::Compiler, vm::{Value, VirtualMachine}};
//!
//! let mut vm = VirtualMachine::new();
//!
//! let source = "2 * (5 - 2);";
//! let mut compiler = Compiler::from(source);
//! let ops = compiler.compile().unwrap();
//!
//! // need to make sure the compiler output is loaded into the vm before calling `next`.
//! vm.load(ops.iter());
//!
//! assert_eq!(vm.next(), Some(Ok(Value::Number(6.0))));
//! ```

#[cfg(test)]
#[macro_use]
extern crate assert_matches;

pub mod ast;
pub mod compiler;
pub mod parser;
pub mod scanner;
pub mod vm;
