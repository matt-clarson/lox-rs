use std::{
    collections::HashSet,
    error::Error,
    fmt::{Debug, Display},
    io::{self, Write},
    mem,
    rc::Rc,
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
///     Op::Print(Span::default()),
///     Op::Return,
/// ];
///
/// let mut buf: Vec<u8> = vec![];
/// let mut vm = VirtualMachine::new(&mut buf);
/// vm.load(instructions);
///
/// assert_eq!(vm.exec(), Ok(()));
/// assert_eq!(String::from_utf8(buf), Ok(String::from("10\n")));
/// ```
pub struct VirtualMachine<I: Iterator<Item = Op>, W: Write> {
    stdout: W,
    ops: Option<I>,
    stack: Stack,
    strings: HashSet<Rc<str>>,
    debug: bool,
}

impl<I: Iterator<Item = Op>> Default for VirtualMachine<I, io::Stdout> {
    fn default() -> Self {
        Self::new(io::stdout())
    }
}

impl<I: Iterator<Item = Op>, W: Write> VirtualMachine<I, W> {
    pub fn new(stdout: W) -> Self {
        Self {
            stdout,
            ops: None,
            stack: Stack::new(),
            strings: HashSet::new(),
            debug: false,
        }
    }

    /// Loads a new set of operations into the VM.
    pub fn load<T: IntoIterator<IntoIter = I>>(&mut self, ops: T) {
        self.ops = Some(ops.into_iter());
    }

    /// Execute the currently loaded operations.
    pub fn exec(&mut self) -> Result<(), VmError> {
        while let Some(op) = self.ops.as_mut().and_then(Iterator::next) {
            if self.debug {
                self.print_debug(&op);
            }

            let r = self.process_op(op);

            if let Ok(Some(_)) = r {
                return Ok(());
            } else if let Err(err) = r {
                return Err(err);
            }
        }

        Ok(())
    }

    fn process_op(&mut self, op: Op) -> Result<Option<()>, VmError> {
        match op {
            Op::Constant(value) => self.push_value(value),
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
                Some(Value::Nil) => Err(VmError::Type(Type::Bool, Type::Nil)),
                Some(Value::Obj(Obj::String(_))) => Err(VmError::Type(Type::Bool, Type::String)),
                None => Err(VmError::NoValue),
            },
            Op::Negate(_) => self.modify_number(|n| -n),
            Op::Add(_) => match self.stack.peek() {
                Some(Value::Number(_)) => self.binary_op_number(|a, b| a + b),
                Some(Value::Obj(Obj::String(_))) => {
                    self.binary_op_string(|a, b| (String::from(a) + b).into())
                }
                Some(Value::True | Value::False) => Err(VmError::Type(Type::Number, Type::Bool)),
                Some(Value::Nil) => Err(VmError::Type(Type::Number, Type::Nil)),
                None => Err(VmError::NoValue),
            },
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
                .and_then(|(v1, v2)| self.push_bool(self.values_eq(&v1, &v2))),
            Op::NotEquals(_) => self
                .pop_two_values()
                .and_then(|(v1, v2)| self.push_bool(!self.values_eq(&v1, &v2))),
            Op::Return => Ok(Some(())),
            Op::Pop => self.stack.pop().ok_or(VmError::NoValue).and(Ok(None)),
            Op::Print(_) => {
                let v = self.stack.pop().ok_or(VmError::NoValue)?;
                if self.debug {
                    self.stdout.write(b"out= ")?;
                }
                self.stdout.write_all(v.to_string().as_bytes())?;
                self.stdout.write(b"\n")?;
                Ok(None)
            }
        }
    }

    fn values_eq(&self, a: &Value, b: &Value) -> bool {
        // because we're interning strings in the self.strings HashSet, we can use pointer equality
        // for fast string comparisons, rather than the default PartialEq implementation which
        // compares by value (the `values` in this function name refers to the `Value` enum, which
        // is confusing but idk naming is hard).
        // We use a method here rather than providing a custom PartialEq implementation because
        // comparing by value is still useful for e.g. unit tests.
        match a {
            Value::Obj(Obj::String(s_a)) => match b {
                Value::Obj(Obj::String(s_b)) => Rc::ptr_eq(s_a, s_b),
                _ => false,
            },
            _ => a == b,
        }
    }

    fn intern_string(&mut self, s: Rc<str>) -> Rc<str> {
        if let Some(interned) = self.strings.get(&s) {
            Rc::clone(interned)
        } else {
            self.strings.insert(Rc::clone(&s));
            s
        }
    }

    fn push_value(&mut self, value: Value) -> Result<Option<()>, VmError> {
        match value {
            Value::Obj(Obj::String(s)) => self.push_string(s),
            v => {
                self.stack.push(v);
                Ok(None)
            }
        }
    }

    fn push_string(&mut self, s: Rc<str>) -> Result<Option<()>, VmError> {
        let interned = self.intern_string(s);
        self.stack.push(interned.into());
        Ok(None)
    }

    fn push_bool(&mut self, b: bool) -> Result<Option<()>, VmError> {
        self.stack.push(if b { Value::True } else { Value::False });
        Ok(None)
    }

    fn binary_op_number<F: FnOnce(f64, f64) -> f64>(
        &mut self,
        f: F,
    ) -> Result<Option<()>, VmError> {
        self.pop_number()
            .and_then(|a| self.modify_number(|b| f(b, a)))
    }

    fn binary_op_string<F: FnOnce(&str, &str) -> Rc<str>>(
        &mut self,
        f: F,
    ) -> Result<Option<()>, VmError> {
        self.pop_two_strings()
            .and_then(|(a, b)| self.push_string(f(&a, &b)))
    }

    fn pop_two_numbers(&mut self) -> Result<(f64, f64), VmError> {
        self.pop_number()
            .and_then(|b| self.pop_number().map(|a| (a, b)))
    }

    fn pop_two_strings(&mut self) -> Result<(Rc<str>, Rc<str>), VmError> {
        self.pop_string()
            .and_then(|b| self.pop_string().map(|a| (a, b)))
    }

    fn pop_two_values(&mut self) -> Result<(Value, Value), VmError> {
        self.pop_value()
            .and_then(|v2| self.pop_value().map(|v1| (v1, v2)))
    }

    fn pop_number(&mut self) -> Result<f64, VmError> {
        match self.stack.pop() {
            Some(Value::Number(n)) => Ok(n),
            Some(Value::True | Value::False) => Err(VmError::Type(Type::Number, Type::Bool)),
            Some(Value::Nil) => Err(VmError::Type(Type::Number, Type::Nil)),
            Some(Value::Obj(Obj::String(_))) => Err(VmError::Type(Type::Number, Type::String)),
            None => Err(VmError::NoValue),
        }
    }

    fn pop_string(&mut self) -> Result<Rc<str>, VmError> {
        match self.stack.pop() {
            Some(Value::Obj(Obj::String(s))) => Ok(s),
            Some(Value::True | Value::False) => Err(VmError::Type(Type::String, Type::Bool)),
            Some(Value::Nil) => Err(VmError::Type(Type::String, Type::Nil)),
            Some(Value::Number(_)) => Err(VmError::Type(Type::String, Type::Number)),
            None => Err(VmError::NoValue),
        }
    }

    fn pop_value(&mut self) -> Result<Value, VmError> {
        match self.stack.pop() {
            Some(v) => Ok(v),
            None => Err(VmError::NoValue),
        }
    }

    fn modify_number<F: FnOnce(f64) -> f64>(&mut self, f: F) -> Result<Option<()>, VmError> {
        match self.stack.peek_mut() {
            Some(Value::Number(n)) => {
                let _ = mem::replace(n, f(*n));
                Ok(None)
            }
            Some(Value::True | Value::False) => Err(VmError::Type(Type::Number, Type::Bool)),
            Some(Value::Nil) => Err(VmError::Type(Type::Number, Type::Nil)),
            Some(Value::Obj(Obj::String(_))) => Err(VmError::Type(Type::Number, Type::String)),
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
    /// Pops the top value from the stack and discards it.
    Pop,
    /// Pops the top value from the stack and sends it to stdout.
    Print(Span),
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
            Self::Return => f.write_str("OP_RETURN"),
            Self::Pop => f.write_str("OP_POP"),
            Self::Print(t) => write!(f, "OP_PRINT ({t})"),
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

    fn peek(&self) -> Option<&Value> {
        self.values.last()
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
#[derive(PartialEq)]
pub enum Value {
    /// A numerical, floating-point value.
    Number(f64),
    /// A boolean true.
    True,
    /// A boolean false.
    False,
    /// A nil value.
    Nil,
    /// An object - value is stored on the heap.
    Obj(Obj),
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::Obj(Obj::String(value.into()))
    }
}

impl From<Rc<str>> for Value {
    fn from(value: Rc<str>) -> Self {
        Self::Obj(Obj::String(value))
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "NUM  {n}"),
            Self::True => f.write_str("BOOL true"),
            Self::False => f.write_str("BOOL false"),
            Self::Nil => f.write_str("NIL"),
            Self::Obj(o) => write!(f, "{o:?}"),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::True => f.write_str("true"),
            Self::False => f.write_str("false"),
            Self::Nil => f.write_str("nil"),
            Self::Obj(o) => write!(f, "{o}"),
        }
    }
}

/// Represents heap allocated values.
#[derive(PartialEq)]
pub enum Obj {
    String(Rc<str>),
}

impl Debug for Obj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "STR  \"{s}\""),
        }
    }
}

impl Display for Obj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "\"{s}\""),
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
    Io(Box<str>),
}

impl From<io::Error> for VmError {
    fn from(value: io::Error) -> Self {
        Self::Io(value.to_string().into())
    }
}

/// An abstract representation of the types a [`Value`] can take. Used only for error reporting.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Type {
    Number,
    Bool,
    Nil,
    String,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number => f.write_str("number"),
            Self::Bool => f.write_str("boolean"),
            Self::Nil => f.write_str("nil"),
            Self::String => f.write_str("string"),
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
            Self::Io(s) => write!(f, "io error: {s}"),
        }
    }
}

impl Error for VmError {}

#[cfg(test)]
mod test {
    use crate::scanner::Span;

    use super::*;

    struct TestWriter {
        b: Vec<u8>,
    }

    impl TestWriter {
        fn new() -> Self {
            Self { b: Vec::new() }
        }
    }

    impl Display for TestWriter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(String::from_utf8(self.b.clone()).unwrap().as_str())
        }
    }

    impl io::Write for TestWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.b.write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.b.flush()
        }
    }

    #[test]
    fn push_and_pop_number() {
        let instructions = vec![
            Op::Constant(Value::Number(6.4)),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();

        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "6.4\n");
    }

    #[test]
    fn not() {
        let instructions = vec![
            Op::Constant(Value::True),
            Op::Not(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::False),
            Op::Not(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "false\ntrue\n");
    }

    #[test]
    fn not_number() {
        let instructions = vec![
            Op::Constant(Value::Number(1.0)),
            Op::Not(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Bool, Type::Number)));
    }

    #[test]
    fn negate_number() {
        let instructions = vec![
            Op::Constant(Value::Number(1.0)),
            Op::Negate(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "-1\n");
    }

    #[test]
    fn negate_bool() {
        let instructions = vec![
            Op::Constant(Value::True),
            Op::Negate(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn addition() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::Number(8.7)),
            Op::Add(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "10\n");
    }

    #[test]
    fn string_concatenation() {
        let instructions = vec![
            Op::Constant("hello".into()),
            Op::Constant(" ".into()),
            Op::Add(Span::default()),
            Op::Constant("world".into()),
            Op::Add(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "\"hello world\"\n");
    }

    #[test]
    fn add_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Add(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn subtraction() {
        let instructions = vec![
            Op::Constant(Value::Number(5.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Subtract(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "1\n");
    }

    #[test]
    fn subtract_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Subtract(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn multiplication() {
        let instructions = vec![
            Op::Constant(Value::Number(5.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Multiply(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "20\n");
    }

    #[test]
    fn multiply_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Multiply(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn division() {
        let instructions = vec![
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Divide(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "5\n");
    }

    #[test]
    fn divide_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Divide(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn compare_less() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::LessThan(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::LessThan(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::LessThan(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "false\nfalse\ntrue\n");
    }

    #[test]
    fn less_than_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::LessThan(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn compare_less_or_equal() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::LessThanOrEqual(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::LessThanOrEqual(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::LessThanOrEqual(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "false\ntrue\ntrue\n");
    }

    #[test]
    fn less_than_or_equal_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::LessThanOrEqual(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn compare_greater() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::GreaterThan(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::GreaterThan(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::GreaterThan(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "true\nfalse\nfalse\n");
    }

    #[test]
    fn greater_than_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::GreaterThan(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn compare_greater_or_equal() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::GreaterThanOrEqual(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::GreaterThanOrEqual(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::GreaterThanOrEqual(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(test_writer.to_string(), "true\ntrue\nfalse\n");
    }

    #[test]
    fn greater_than_or_equal_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::GreaterThanOrEqual(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Err(VmError::Type(Type::Number, Type::Bool)));
    }

    #[test]
    fn equals() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Equals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(20.0)),
            Op::Equals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::True),
            Op::Constant(Value::True),
            Op::Equals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::False),
            Op::Constant(Value::False),
            Op::Equals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::True),
            Op::Constant(Value::False),
            Op::Equals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant("hello".into()),
            Op::Constant("hello".into()),
            Op::Equals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant("hello".into()),
            Op::Constant("world".into()),
            Op::Equals(Span::default()),
            Op::Print(Span::default()),
            Op::Return,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);

        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(
            test_writer.to_string(),
            "true\nfalse\ntrue\ntrue\nfalse\ntrue\nfalse\n"
        );
    }
    
    #[test]
    fn not_equals() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(4.0)),
            Op::NotEquals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(20.0)),
            Op::NotEquals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::True),
            Op::Constant(Value::True),
            Op::NotEquals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::False),
            Op::Constant(Value::False),
            Op::NotEquals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant(Value::True),
            Op::Constant(Value::False),
            Op::NotEquals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant("hello".into()),
            Op::Constant("hello".into()),
            Op::NotEquals(Span::default()),
            Op::Print(Span::default()),
            Op::Constant("hello".into()),
            Op::Constant("world".into()),
            Op::NotEquals(Span::default()),
            Op::Print(Span::default()),
            Op::Return
        ];
    
        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);
        vm.load(instructions);
    
        assert_eq!(vm.exec(), Ok(()));
        assert_eq!(
            test_writer.to_string(),
            "false\ntrue\nfalse\nfalse\ntrue\nfalse\ntrue\n"
        );
    }
}
