use lox::{compiler::Compiler, vm::VirtualMachine};

fn main() {
    let mut vm = VirtualMachine::new();
    vm.enable_debug();

    let source = "4 * (8 + 2);";

    let mut compiler = Compiler::from(source);

    let instructions = compiler.compile().unwrap();
    vm.load(&instructions);

    for result in vm {
        match result {
            Ok(value) => println!("got value: {value}"),
            Err(e) => eprintln!("error: {e}"),
        }
    }
}
