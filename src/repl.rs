use std::{error::Error, io, vec::IntoIter};

use rustyline::{error::ReadlineError, DefaultEditor};

use crate::{
    compiler::Compiler,
    vm::{Op, VirtualMachine},
    Config,
};

/// Read, Execute, Print, Loop implementation for Lox.
pub struct Repl {
    compiler: Compiler,
    vm: VirtualMachine<IntoIter<Op>, io::Stdout>,
    editor: DefaultEditor,
}

impl Repl {
    pub fn new(config: &Config) -> Self {
        let mut vm = VirtualMachine::default();
        if config.debug {
            vm.enable_debug();
        }

        Self {
            compiler: Compiler::default(),
            vm,
            editor: DefaultEditor::new().unwrap(),
        }
    }

    /// Convenience method for instantiating and starting th loop.
    pub fn run(config: &Config) -> Result<(), Box<dyn Error>> {
        let mut repl = Self::new(config);
        repl.start()
    }

    /// Starts the loop - will only exit on an unrecoverable error.
    ///
    /// Parse, compile, and runtime errors will not halt the loop.
    pub fn start(&mut self) -> Result<(), Box<dyn Error>> {
        loop {
            let readline = self.editor.readline(">> ");
            match readline {
                Ok(line) => {
                    self.editor.add_history_entry(line.as_str())?;

                    let ops = match self.compiler.compile(line.as_str()) {
                        Ok(ops) => ops,
                        Err(err) => {
                            println!("Error: {err}");
                            continue;
                        }
                    };

                    self.vm.load(ops);
                    if let Err(err) = self.vm.exec() {
                        println!("Error: {err}");
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("Quitting.");
                    break;
                }
                Err(err) => return Err(err.into()),
            }
        }

        Ok(())
    }
}
