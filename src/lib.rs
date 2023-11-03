//! Contains the scanner, parser, compiler, and virtual machine used to interpret and execute Lox source code.
//!
//! For general usage, the [compiler](crate::compiler::Compiler) will handle instantiating scanning
//! and parsing code. See the example below:
//!
//! ```
//! use lox::{compiler::Compiler, vm::{Value, VirtualMachine}};
//!
//! let source = "2 * (5 - 2);";
//!
//! let mut compiler = Compiler::from(source);
//!
//! let ops = compiler.compile().unwrap();
//!
//! let mut vm = VirtualMachine::new();
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
