use std::{
    error::Error,
    fmt::{Debug, Display},
    mem,
};

use crate::scanner::Span;

/// Stack-based VM for executing compiled Lox bytecode.
///
///See the [crate docs](crate) for docs on using the VM with a compiler.
///
/// If the VM encounters an error it will return it and will then halt execution, even if the
/// provided set of instructions would otherwise continue past the error point.
///
/// Use [`enable_debug`](crate::vm::VirtualMachine::enable_debug) to turn on a debugger which
/// will print the full state of the internal stack before processing each operation.
///
/// # Example
///
/// ```
/// use lox::vm::*;
/// use lox::scanner::Span;
///
/// let instructions = vec![
///     Op::Constant(Value::Number(1.3)),
///     Op::Constant(Value::Number(8.7)),
///     Op::Add(Span::default()),
///     Op::Return,
/// ];
///
/// let mut vm = VirtualMachine::new();
/// vm.load(instructions);
///
/// assert_eq!(vm.next(), Some(Ok(Value::Number(10.0))));
/// assert_eq!(vm.next(), None);
/// ```
pub struct VirtualMachine<I: Iterator<Item = Op>> {
    ops: Option<I>,
    stack: Stack,
    errored: bool,
    debug: bool,
}

impl<I: Iterator<Item = Op>> Default for VirtualMachine<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Iterator<Item = Op>> Iterator for VirtualMachine<I> {
    type Item = Result<Value, VmError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.errored {
            return None;
        }

        while let Some(op) = self.ops.as_mut()?.next() {
            if self.debug {
                self.print_debug(&op);
            }

            let r = self.process_op(op);

            if let Ok(Some(_)) = r {
                return r.transpose();
            } else if r.is_err() {
                self.errored = true;
                return r.transpose();
            }
        }

        None
    }
}

impl<I: Iterator<Item = Op>> VirtualMachine<I> {
    pub fn new() -> Self {
        Self {
            ops: None,
            stack: Stack::new(),
            errored: false,
            debug: false,
        }
    }

    /// Loads a new set of operations into the VM. These can then be executed using the VM
    /// [Iterator] implementation.
    pub fn load<T: IntoIterator<IntoIter = I>>(&mut self, ops: T) {
        self.ops = Some(ops.into_iter());
    }

    fn process_op(&mut self, op: Op) -> Result<Option<Value>, VmError> {
        match op {
            Op::Constant(value) => {
                self.stack.push(value);
                Ok(None)
            }
            Op::Not(_) => match self.stack.peek_mut() {
                Some(v @ Value::True) => {
                    let _ = mem::replace(v, Value::False);
                    Ok(None)
                }
                Some(v @ Value::False) => {
                    let _ = mem::replace(v, Value::True);
                    Ok(None)
                }
                Some(Value::Number(_)) => Err(VmError::Type(Type::Bool, Type::Number)),
                None => Err(VmError::NoValue),
            },
            Op::Negate(_) => self.modify_number(|n| -n),
            Op::Add(_) => self.binary_op_number(|a, b| a + b),
            Op::Subtract(_) => self.binary_op_number(|a, b| a - b),
            Op::Multiply(_) => self.binary_op_number(|a, b| a * b),
            Op::Divide(_) => self.binary_op_number(|a, b| a / b),
            Op::LessThan(_) => self
                .pop_two_numbers()
                .and_then(|(a, b)| self.push_bool(a < b)),
            Op::LessThanOrEqual(_) => self
                .pop_two_numbers()
                .and_then(|(a, b)| self.push_bool(a <= b)),
            Op::GreaterThan(_) => self
                .pop_two_numbers()
                .and_then(|(a, b)| self.push_bool(a > b)),
            Op::GreaterThanOrEqual(_) => self
                .pop_two_numbers()
                .and_then(|(a, b)| self.push_bool(a >= b)),
            Op::Equals(_) => self
                .pop_two_values()
                .and_then(|(v1, v2)| self.push_bool(v1 == v2)),
            Op::NotEquals(_) => self
                .pop_two_values()
                .and_then(|(v1, v2)| self.push_bool(v1 != v2)),
            Op::Return => self.stack.pop().ok_or(VmError::NoValue).map(Some),
        }
    }

    fn binary_op_number<F: FnOnce(f64, f64) -> f64>(
        &mut self,
        f: F,
    ) -> Result<Option<Value>, VmError> {
        self.pop_number()
            .and_then(|a| self.modify_number(|b| f(b, a)))
    }

    fn push_bool(&mut self, b: bool) -> Result<Option<Value>, VmError> {
        self.stack.push(if b { Value::True } else { Value::False });
        Ok(None)
    }

    fn pop_two_numbers(&mut self) -> Result<(f64, f64), VmError> {
        self.pop_number()
            .and_then(|b| self.pop_number().map(|a| (a, b)))
    }

    fn pop_two_values(&mut self) -> Result<(Value, Value), VmError> {
        self.pop_value()
            .and_then(|v2| self.pop_value().map(|v1| (v1, v2)))
    }

    fn pop_number(&mut self) -> Result<f64, VmError> {
        match self.stack.pop() {
            Some(Value::Number(n)) => Ok(n),
            Some(Value::True | Value::False) => Err(VmError::Type(Type::Number, Type::Bool)),
            None => Err(VmError::NoValue),
        }
    }

    fn pop_value(&mut self) -> Result<Value, VmError> {
        match self.stack.pop() {
            Some(v) => Ok(v),
            None => Err(VmError::NoValue),
        }
    }

    fn modify_number<F: FnOnce(f64) -> f64>(&mut self, f: F) -> Result<Option<Value>, VmError> {
        match self.stack.peek_mut() {
            Some(Value::Number(n)) => {
                let _ = mem::replace(n, f(*n));
                Ok(None)
            }
            Some(Value::True | Value::False) => Err(VmError::Type(Type::Number, Type::Bool)),
            None => Err(VmError::NoValue),
        }
    }

    /// Enables debugging, printing to stdout before processing each op code.
    ///
    /// Note that the debugger is _disabled_ by default.
    pub fn enable_debug(&mut self) {
        self.debug = true;
    }

    /// Disables the debugger.
    pub fn disable_debug(&mut self) {
        self.debug = false;
    }

    fn print_debug(&self, op: &Op) {
        println!("-- vm debug --");
        println!("  NEXT OP: {op}");
        println!("  -- stack ({}) --", self.stack.size());
        println!("{}", self.stack);
    }
}

/// Bytecode operations used by the [VM](VirtualMachine).
#[derive(Debug, PartialEq)]
pub enum Op {
    /// Pops the top value from the stack to be returned to the caller.
    Return,
    /// Pushes the wrapped [`Value`] onto the stack.
    Constant(Value),
    /// Flips the top value on the stack, if it is a boolean.
    Not(Span),
    /// Negates the top number on the stack.
    Negate(Span),
    /// Takes the top two numbers on the stack and pushes the sum onto the stack.
    Add(Span),
    /// Takes the top two numbers on the stack and pushes the difference onto the stack.
    Subtract(Span),
    /// Takes the top two numbers on the stack and pushes the product onto the stack.
    Multiply(Span),
    /// Takes the top two numbers on the stack and pushes the quotient onto the stack.
    Divide(Span),
    /// Takes the top two numbers and pushes true onto the stack if the first is less than the
    /// second, otherwise pushes false.
    LessThan(Span),
    /// Takes the top two numbers and pushes true onto the stack if the first is less than or equal
    /// to the second, otherwise pushes false.
    LessThanOrEqual(Span),
    /// Takes the top two numbers and pushes true onto the stack if the first is greater than the
    /// second, otherwise pushes false.
    GreaterThan(Span),
    /// Takes the top two numbers and pushes true onto the stack if the first is greater than or
    /// equal to the second, otherwise pushes false.
    GreaterThanOrEqual(Span),
    /// Takes the top two values and pushes true onto the stack if they are equal, otherwise false.
    /// Values of different types are _never_ equal to one another.
    Equals(Span),
    /// Takes the top two values and pushes true onto the stack if they are not equal, otherwise
    /// false. Values of different types are _never_ equal to one another.
    NotEquals(Span),
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Return => f.write_str("OP_RETURN"),
            Self::Constant(v) => write!(f, "OP_CONSTANT {v:?}"),
            Self::Not(t) => write!(f, "OP_NOT ({t})"),
            Self::Negate(t) => write!(f, "OP_NEG ({t})"),
            Self::Add(t) => write!(f, "OP_ADD ({t})"),
            Self::Subtract(t) => write!(f, "OP_SUB ({t})"),
            Self::Multiply(t) => write!(f, "OP_MULT ({t})"),
            Self::Divide(t) => write!(f, "OP_DIV ({t})"),
            Self::LessThan(t) => write!(f, "OP_LT ({t})"),
            Self::LessThanOrEqual(t) => write!(f, "OP_LTE ({t})"),
            Self::GreaterThan(t) => write!(f, "OP_GT ({t})"),
            Self::GreaterThanOrEqual(t) => write!(f, "OP_GTE ({t})"),
            Self::Equals(t) => write!(f, "OP_EQ ({t})"),
            Self::NotEquals(t) => write!(f, "OP_NEQ ({t})"),
        }
    }
}

struct Stack {
    values: Vec<Value>,
}

impl Stack {
    fn new() -> Self {
        Self { values: vec![] }
    }

    fn push(&mut self, value: Value) {
        self.values.push(value)
    }

    fn pop(&mut self) -> Option<Value> {
        self.values.pop()
    }

    fn peek_mut(&mut self) -> Option<&mut Value> {
        self.values.last_mut()
    }

    fn size(&self) -> usize {
        self.values.len()
    }
}

impl Display for Stack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, value) in self.values.iter().enumerate().rev() {
            writeln!(f, "    [{i:04}\t{value:?}]")?;
        }
        Ok(())
    }
}

/// Represents a storable value, which can be moved on and off the [VM](VirtualMachine)'s internal
/// stack.
#[derive(Copy, Clone, PartialEq)]
pub enum Value {
    /// A numerical, floating-point value.
    Number(f64),
    /// A boolean true.
    True,
    /// A boolean false.
    False,
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "NUM  {n}"),
            Self::True => f.write_str("BOOL true"),
            Self::False => f.write_str("BOOL false"),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::True => f.write_str("true"),
            Self::False => f.write_str("false"),
        }
    }
}

/// Represents errors that can be produced by the [VM](VirtualMachine).
#[derive(Debug, PartialEq)]
pub enum VmError {
    /// A specific type of runtime error, which occurs when the VM tries to pop from an empty
    /// stack. Should always be considered a fatal error.
    NoValue,
    /// A generic runtime error.
    Runtime(Box<str>),
    /// A type error, where an operation expects a value of a specific type, but the next value on
    /// the stack has a different type. The first type here is the expected one, the second the
    /// actual type encountered.
    Type(Type, Type),
}

/// An abstract representation of the types a [`Value`] can take. Used only for error reporting.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Type {
    Number,
    Bool,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number => f.write_str("number"),
            Self::Bool => f.write_str("boolean"),
        }
    }
}

impl Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoValue => write!(f, "runtime error: no value on stack"),
            Self::Runtime(s) => write!(f, "runtime error: {s}"),
            Self::Type(expected, actual) => {
                write!(f, "type error: expected {expected}, got {actual}")
            }
        }
    }
}

impl Error for VmError {}

#[cfg(test)]
mod test {
    use crate::scanner::Span;

    use super::*;

    #[test]
    fn push_and_pop_number() {
        let instructions = vec![Op::Constant(Value::Number(6.4)), Op::Return];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::Number(6.4))));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn not() {
        let instructions = vec![
            Op::Constant(Value::True),
            Op::Not(Span::default()),
            Op::Return,
            Op::Constant(Value::False),
            Op::Not(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn not_number() {
        let instructions = vec![
            Op::Constant(Value::Number(1.0)),
            Op::Not(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Bool, Type::Number)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn negate_number() {
        let instructions = vec![
            Op::Constant(Value::Number(1.0)),
            Op::Negate(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::Number(-1.0))));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn negate_bool() {
        let instructions = vec![
            Op::Constant(Value::True),
            Op::Negate(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn addition() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::Number(8.7)),
            Op::Add(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::Number(10.0))));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn add_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Add(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn subtraction() {
        let instructions = vec![
            Op::Constant(Value::Number(5.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Subtract(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::Number(1.0))));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn subtract_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Subtract(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn multiplication() {
        let instructions = vec![
            Op::Constant(Value::Number(5.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Multiply(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::Number(20.0))));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn multiply_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Multiply(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn division() {
        let instructions = vec![
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Divide(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::Number(5.0))));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn divide_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Divide(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn compare_less() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::LessThan(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::LessThan(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::LessThan(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn less_than_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::LessThan(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn compare_less_or_equal() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::LessThanOrEqual(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::LessThanOrEqual(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::LessThanOrEqual(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn less_than_or_equal_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::LessThanOrEqual(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn compare_greater() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::GreaterThan(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::GreaterThan(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::GreaterThan(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn greater_than_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::GreaterThan(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn compare_greater_or_equal() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::GreaterThanOrEqual(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::GreaterThanOrEqual(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::GreaterThanOrEqual(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn greater_than_or_equal_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::GreaterThanOrEqual(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(
            vm.next(),
            Some(Err(VmError::Type(Type::Number, Type::Bool)))
        );
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn equals() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Equals(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(20.0)),
            Op::Equals(Span::default()),
            Op::Return,
            Op::Constant(Value::True),
            Op::Constant(Value::True),
            Op::Equals(Span::default()),
            Op::Return,
            Op::Constant(Value::False),
            Op::Constant(Value::False),
            Op::Equals(Span::default()),
            Op::Return,
            Op::Constant(Value::True),
            Op::Constant(Value::False),
            Op::Equals(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), None);
    }

    #[test]
    fn not_equals() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(4.0)),
            Op::NotEquals(Span::default()),
            Op::Return,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(20.0)),
            Op::NotEquals(Span::default()),
            Op::Return,
            Op::Constant(Value::True),
            Op::Constant(Value::True),
            Op::NotEquals(Span::default()),
            Op::Return,
            Op::Constant(Value::False),
            Op::Constant(Value::False),
            Op::NotEquals(Span::default()),
            Op::Return,
            Op::Constant(Value::True),
            Op::Constant(Value::False),
            Op::NotEquals(Span::default()),
            Op::Return,
        ];

        let mut vm = VirtualMachine::new();
        vm.load(instructions);

        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::False)));
        assert_eq!(vm.next(), Some(Ok(Value::True)));
        assert_eq!(vm.next(), None);
    }
}
