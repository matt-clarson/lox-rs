use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt::{Debug, Display},
    io::{self, Write},
    mem,
    rc::Rc,
};

use crate::compiler::{Chunk, ChunkIter};

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
///
/// let instructions = vec![
///     Op::Constant(Value::Number(1.3)),
///     Op::Constant(Value::Number(8.7)),
///     Op::Add,
///     Op::Print,
/// ];
///
/// let mut buf: Vec<u8> = vec![];
/// let mut vm = VirtualMachine::new(&mut buf);
///
///
/// assert_eq!(vm.exec(instructions.into()), Ok(()));
/// assert_eq!(String::from_utf8(buf), Ok(String::from("10\n")));
/// ```
pub struct VirtualMachine<W: Write> {
    stdout: W,
    stack: Stack,
    call_stack: Vec<CallFrame>,
    globals: HashMap<Rc<str>, Value>,
    strings: HashSet<Rc<str>>,
    debug: bool,
}

impl Default for VirtualMachine<io::Stdout> {
    fn default() -> Self {
        Self::new(io::stdout())
    }
}

impl<W: Write> VirtualMachine<W> {
    pub fn new(stdout: W) -> Self {
        Self {
            stdout,
            stack: Stack::new(),
            call_stack: Vec::new(),
            globals: HashMap::new(),
            strings: HashSet::new(),
            debug: false,
        }
    }

    /// Return the current call stack - useful to provide context on runtime errors.
    pub fn call_stack(&self) -> Box<[&str]> {
        self.call_stack
            .iter()
            .map(|f| f.name.as_ref())
            .collect::<Vec<&str>>()
            .into()
    }

    /// Execute the currently loaded operations.
    pub fn exec(&mut self, chunk: Chunk) -> Result<(), VmError> {
        let ops = chunk.iter();
        self.exec_ops(ops)?;
        Ok(())
    }

    fn exec_ops(&mut self, ops: ChunkIter<'_>) -> Result<Value, VmError> {
        for op in ops {
            if self.debug {
                self.print_debug(op);
            }

            let r = self.process_op(op);

            if let Ok(Some(value)) = r {
                return Ok(value);
            } else if let Err(err) = r {
                return Err(err);
            }
        }

        Ok(Value::Nil)
    }

    fn process_op(&mut self, op: &Op) -> Result<Option<Value>, VmError> {
        match op {
            Op::Constant(value) => self.push_value(value.clone()),
            Op::Not => match self.stack.peek_mut() {
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
                Some(Value::Obj(Obj::Function(_))) => {
                    Err(VmError::Type(Type::Bool, Type::Function))
                }
                None => Err(VmError::NoValue),
            },
            Op::Negate => self.modify_number(|n| -n),
            Op::Add => match self.stack.peek() {
                Some(Value::Number(_)) => self.binary_op_number(|a, b| a + b),
                Some(Value::Obj(Obj::String(_))) => {
                    self.binary_op_string(|a, b| (String::from(a) + b).into())
                }
                Some(Value::True | Value::False) => Err(VmError::Type(Type::Number, Type::Bool)),
                Some(Value::Nil) => Err(VmError::Type(Type::Number, Type::Nil)),
                Some(Value::Obj(Obj::Function(_))) => {
                    Err(VmError::Type(Type::Number, Type::Function))
                }
                None => Err(VmError::NoValue),
            },
            Op::Subtract => self.binary_op_number(|a, b| a - b),
            Op::Multiply => self.binary_op_number(|a, b| a * b),
            Op::Divide => self.binary_op_number(|a, b| a / b),
            Op::LessThan => self
                .pop_two_numbers()
                .and_then(|(a, b)| self.push_bool(a < b)),
            Op::LessThanOrEqual => self
                .pop_two_numbers()
                .and_then(|(a, b)| self.push_bool(a <= b)),
            Op::GreaterThan => self
                .pop_two_numbers()
                .and_then(|(a, b)| self.push_bool(a > b)),
            Op::GreaterThanOrEqual => self
                .pop_two_numbers()
                .and_then(|(a, b)| self.push_bool(a >= b)),
            Op::Equals => self
                .pop_two_values()
                .and_then(|(v1, v2)| self.push_bool(self.values_eq(&v1, &v2))),
            Op::NotEquals => self
                .pop_two_values()
                .and_then(|(v1, v2)| self.push_bool(!self.values_eq(&v1, &v2))),
            Op::Return => self.pop_value().map(Some),
            Op::VoidReturn => Ok(Some(Value::Nil)),
            Op::Pop => self.stack.pop().ok_or(VmError::NoValue).and(Ok(None)),
            Op::PopN(n) => {
                self.stack.popn(*n);
                Ok(None)
            }
            Op::Print => {
                let v = self.stack.pop().ok_or(VmError::NoValue)?;
                if self.debug {
                    self.stdout.write_all(b"out= ")?;
                }
                self.stdout.write_all(v.to_string().as_bytes())?;
                self.stdout.write_all(b"\n")?;
                Ok(None)
            }
            Op::DefineGlobal(ident) => {
                let interned = unsafe { self.intern_string(ident.as_ref_str()) };
                let value = self.pop_value()?;
                self.globals.insert(interned, value);
                Ok(None)
            }
            Op::GetGlobal(ident) => {
                let s = unsafe { ident.as_ref_str() };
                if let Some(value) = self.globals.get(&s) {
                    self.push_value(value.clone())
                } else {
                    Err(VmError::Undefined(s.as_ref().into()))
                }
            }
            Op::SetGlobal(ident) => {
                let s = unsafe { ident.as_ref_str() };
                if self.globals.get(&s).is_none() {
                    return Err(VmError::Undefined(s.as_ref().into()));
                }
                let new_value = self.clone_value()?;
                self.globals.insert(s, new_value);
                Ok(None)
            }
            Op::GetLocal(idx) => {
                let pos = self.stack_pos(*idx);
                self.stack.copy_to_top(pos);
                Ok(None)
            }
            Op::SetLocal(idx) => {
                let value = self.pop_value()?;
                let pos = self.stack_pos(*idx);
                self.stack.insert(pos, value);
                Ok(None)
            }
            Op::Call(n) => {
                let function = self.pop_function()?;

                if n != &function.arity {
                    return Err(VmError::WrongNumArgs {
                        name: function.name.as_ref().into(),
                        expected: function.arity,
                        actual: *n
                    });
                }

                let offset = self.stack.size() - function.arity as usize;
                let call_frame = CallFrame {
                    name: Rc::clone(&function.name),
                    offset,
                };

                self.call_stack.push(call_frame);

                let return_value = self.exec_ops(function.ops())?;

                self.call_stack.pop();

                self.stack.truncate(offset);

                self.push_value(return_value)
            }
        }
    }

    fn stack_pos(&self, pos: usize) -> usize {
        if let Some(call_frame) = self.call_stack.last() {
            call_frame.offset + pos
        } else {
            pos
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

    fn push_value(&mut self, value: Value) -> Result<Option<Value>, VmError> {
        match value {
            Value::Obj(Obj::String(s)) => self.push_string(s),
            v => {
                self.stack.push(v);
                Ok(None)
            }
        }
    }

    fn push_string(&mut self, s: Rc<str>) -> Result<Option<Value>, VmError> {
        let interned = self.intern_string(s);
        self.stack.push(interned.into());
        Ok(None)
    }

    fn push_bool(&mut self, b: bool) -> Result<Option<Value>, VmError> {
        self.stack.push(if b { Value::True } else { Value::False });
        Ok(None)
    }

    fn binary_op_number<F: FnOnce(f64, f64) -> f64>(
        &mut self,
        f: F,
    ) -> Result<Option<Value>, VmError> {
        self.pop_number()
            .and_then(|a| self.modify_number(|b| f(b, a)))
    }

    fn binary_op_string<F: FnOnce(&str, &str) -> Rc<str>>(
        &mut self,
        f: F,
    ) -> Result<Option<Value>, VmError> {
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
            Some(Value::Obj(Obj::Function(_))) => Err(VmError::Type(Type::Number, Type::Function)),
            None => Err(VmError::NoValue),
        }
    }

    fn pop_string(&mut self) -> Result<Rc<str>, VmError> {
        match self.stack.pop() {
            Some(Value::Obj(Obj::String(s))) => Ok(s),
            Some(Value::Obj(Obj::Function(_))) => Err(VmError::Type(Type::String, Type::Function)),
            Some(Value::True | Value::False) => Err(VmError::Type(Type::String, Type::Bool)),
            Some(Value::Nil) => Err(VmError::Type(Type::String, Type::Nil)),
            Some(Value::Number(_)) => Err(VmError::Type(Type::String, Type::Number)),
            None => Err(VmError::NoValue),
        }
    }

    fn pop_function(&mut self) -> Result<Function, VmError> {
        match self.stack.pop() {
            Some(Value::Obj(Obj::Function(function))) => Ok(function),
            Some(Value::Obj(Obj::String(_))) => Err(VmError::NotCallable(Type::String)),
            Some(Value::True | Value::False) => Err(VmError::NotCallable(Type::Bool)),
            Some(Value::Nil) => Err(VmError::NotCallable(Type::Nil)),
            Some(Value::Number(_)) => Err(VmError::NotCallable(Type::Number)),
            None => Err(VmError::NoValue),
        }
    }

    fn pop_value(&mut self) -> Result<Value, VmError> {
        match self.stack.pop() {
            Some(v) => Ok(v),
            None => Err(VmError::NoValue),
        }
    }

    fn clone_value(&mut self) -> Result<Value, VmError> {
        match self.stack.peek() {
            Some(v) => Ok(v.clone()),
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
            Some(Value::Nil) => Err(VmError::Type(Type::Number, Type::Nil)),
            Some(Value::Obj(Obj::String(_))) => Err(VmError::Type(Type::Number, Type::String)),
            Some(Value::Obj(Obj::Function(_))) => Err(VmError::Type(Type::Number, Type::Function)),
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

struct CallFrame {
    name: Rc<str>,
    offset: usize,
}

/// Bytecode operations used by the [VM](VirtualMachine).
#[derive(Debug, PartialEq)]
pub enum Op {
    /// Pops the top value from the stack to be returned to the caller.
    Return,
    /// Returns nil to the caller. Does not interact with the stack.
    VoidReturn,
    /// Pops the top value from the stack and discards it.
    Pop,
    /// Pops the top n values from the top of the stack and discards them.
    PopN(usize),
    /// Pops the top value from the stack and sends it to stdout.
    Print,
    /// Pushes the wrapped [`Value`] onto the stack.
    Constant(Value),
    /// Flips the top value on the stack, if it is a boolean.
    Not,
    /// Negates the top number on the stack.
    Negate,
    /// Takes the top two numbers on the stack and pushes the sum onto the stack.
    Add,
    /// Takes the top two numbers on the stack and pushes the difference onto the stack.
    Subtract,
    /// Takes the top two numbers on the stack and pushes the product onto the stack.
    Multiply,
    /// Takes the top two numbers on the stack and pushes the quotient onto the stack.
    Divide,
    /// Takes the top two numbers and pushes true onto the stack if the first is less than the
    /// second, otherwise pushes false.
    LessThan,
    /// Takes the top two numbers and pushes true onto the stack if the first is less than or equal
    /// to the second, otherwise pushes false.
    LessThanOrEqual,
    /// Takes the top two numbers and pushes true onto the stack if the first is greater than the
    /// second, otherwise pushes false.
    GreaterThan,
    /// Takes the top two numbers and pushes true onto the stack if the first is greater than or
    /// equal to the second, otherwise pushes false.
    GreaterThanOrEqual,
    /// Takes the top two values and pushes true onto the stack if they are equal, otherwise false.
    /// Values of different types are _never_ equal to one another.
    Equals,
    /// Takes the top two values and pushes true onto the stack if they are not equal, otherwise
    /// false. Values of different types are _never_ equal to one another.
    NotEquals,
    /// Pop the top value from the stack and assign it as a global variable, using the wrapped
    /// [`Value`] as the identifier.
    ///
    /// This operation will overwriting any existing global variable with the same identifier.
    DefineGlobal(Value),
    /// Lookup a global variable using the wrapped [`Value`] as the identifier. Pushes the stored
    /// value onto the stack, or raises an "undefined variable" exception.
    GetGlobal(Value),
    /// If a global variable exists whose identifier matches that represented by the wrapped
    /// [`Value`], replace its value with the top value from the stack, otherwise raise an
    /// "undefined variable" exception.
    ///
    /// This operation does not modify the stack directly - the top value is cloned, not popped.
    SetGlobal(Value),
    /// Access a local variable by index in the VM stack.
    GetLocal(usize),
    /// Pop the top value off the stack, and replace local variable at the given index with the
    /// popped value.
    SetLocal(usize),
    /// Pop the top value off the stack, then call it, creating a new call frame. Errors of the value is not a Function. The wrapped number is the number of args provided by the invoker, used to ensure the same number of args as params are provided.
    Call(u8),
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Return => f.write_str("OP_RETURN"),
            Self::VoidReturn => f.write_str("OP_VOID_RETURN"),
            Self::Pop => f.write_str("OP_POP"),
            Self::PopN(n) => write!(f, "OP_POPN {n}"),
            Self::Print => f.write_str("OP_PRINT"),
            Self::Constant(v) => write!(f, "OP_CONSTANT {v:?}"),
            Self::Not => f.write_str("OP_NOT"),
            Self::Negate => f.write_str("OP_NEG"),
            Self::Add => f.write_str("OP_ADD"),
            Self::Subtract => f.write_str("OP_SUB"),
            Self::Multiply => f.write_str("OP_MULT"),
            Self::Divide => f.write_str("OP_DIV"),
            Self::LessThan => f.write_str("OP_LT"),
            Self::LessThanOrEqual => f.write_str("OP_LTE"),
            Self::GreaterThan => f.write_str("OP_GT"),
            Self::GreaterThanOrEqual => f.write_str("OP_GTE"),
            Self::Equals => f.write_str("OP_EQ"),
            Self::NotEquals => f.write_str("OP_NEQ"),
            Self::DefineGlobal(ident) => write!(f, "OP_DEFG {ident:?}"),
            Self::GetGlobal(ident) => write!(f, "OP_GETG {ident:?}"),
            Self::SetGlobal(ident) => write!(f, "OP_SETG {ident:?}"),
            Self::GetLocal(idx) => write!(f, "OP_GETL {idx}"),
            Self::SetLocal(idx) => write!(f, "OP_SETL {idx}"),
            Self::Call(n) => write!(f, "OP_CALL {n}"),
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

    fn popn(&mut self, n: usize) {
        self.values.truncate(self.values.len() - n);
    }

    fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
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

    fn copy_to_top(&mut self, idx: usize) -> bool {
        if let Some(v) = self.values.get(idx) {
            self.values.push(v.clone());
            true
        } else {
            false
        }
    }

    fn insert(&mut self, idx: usize, value: Value) -> bool {
        if let Some(v) = self.values.get_mut(idx) {
            let _ = mem::replace(v, value);
            true
        } else {
            false
        }
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
#[derive(Clone, PartialEq)]
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

impl Default for Value {
    fn default() -> Self {
        Self::Nil
    }
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

impl From<Function> for Value {
    fn from(value: Function) -> Self {
        Self::Obj(Obj::Function(value))
    }
}

impl Value {
    unsafe fn as_ref_str(&self) -> Rc<str> {
        if let Self::Obj(Obj::String(s)) = self {
            Rc::clone(s)
        } else {
            panic!("value is not a string")
        }
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
    Function(Function),
}

impl Clone for Obj {
    fn clone(&self) -> Self {
        match self {
            Self::String(s) => Self::String(Rc::clone(s)),
            Self::Function(f) => Self::Function(f.clone()),
        }
    }
}

impl Debug for Obj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "STR  \"{s}\""),
            Self::Function(fun) => write!(f, "FUN  {fun:?}"),
        }
    }
}

impl Display for Obj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Function(fun) => write!(f, "{fun}"),
        }
    }
}

/// Represents a function, as a first-class entity.
#[derive(Debug, PartialEq)]
pub struct Function {
    /// Function name for debugging.
    pub name: Rc<str>,
    /// Required number of arguments.
    pub arity: u8,
    /// Function code.
    pub chunk: Rc<[Op]>,
}

impl Function {
    fn ops(&self) -> ChunkIter<'_> {
        ChunkIter::new(&self.chunk)
    }
}

impl Clone for Function {
    fn clone(&self) -> Self {
        Self {
            name: Rc::clone(&self.name),
            arity: self.arity,
            chunk: Rc::clone(&self.chunk),
        }
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{}()>", self.name)
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
    /// An IO error.
    Io(Box<str>),
    /// An error raised by trying to access an undefined variable.
    Undefined(Box<str>),
    /// Raised if trying to call a non-function type.
    NotCallable(Type),
    /// Raised when calling a function with wrong number of arguments.
    WrongNumArgs {
        name: Box<str>,
        expected: u8,
        actual: u8
    },
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
    Function,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number => f.write_str("number"),
            Self::Bool => f.write_str("boolean"),
            Self::Nil => f.write_str("nil"),
            Self::String => f.write_str("string"),
            Self::Function => f.write_str("function"),
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
            Self::Undefined(s) => write!(f, "undefined variable: {s}"),
            Self::NotCallable(t) => write!(f, "not callable: value of type {t} is not callable"),
            Self::WrongNumArgs { name, expected, actual } => {
                write!(f, "wrong number of args: function {name} needs {expected} args, {actual} were provided")
            }
        }
    }
}

impl Error for VmError {}

#[cfg(test)]
mod test {
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
        let instructions = vec![Op::Constant(Value::Number(6.4)), Op::Print];

        let mut test_writer = TestWriter::new();

        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "6.4\n");
    }

    #[test]
    fn not() {
        let instructions = vec![
            Op::Constant(Value::True),
            Op::Not,
            Op::Print,
            Op::Constant(Value::False),
            Op::Not,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "false\ntrue\n");
    }

    #[test]
    fn not_number() {
        let instructions = vec![Op::Constant(Value::Number(1.0)), Op::Not, Op::Print];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Bool, Type::Number))
        );
    }

    #[test]
    fn negate_number() {
        let instructions = vec![Op::Constant(Value::Number(1.0)), Op::Negate, Op::Print];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "-1\n");
    }

    #[test]
    fn negate_bool() {
        let instructions = vec![Op::Constant(Value::True), Op::Negate, Op::Print];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn addition() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::Number(8.7)),
            Op::Add,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "10\n");
    }

    #[test]
    fn string_concatenation() {
        let instructions = vec![
            Op::Constant("hello".into()),
            Op::Constant(" ".into()),
            Op::Add,
            Op::Constant("world".into()),
            Op::Add,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "\"hello world\"\n");
    }

    #[test]
    fn add_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Add,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn subtraction() {
        let instructions = vec![
            Op::Constant(Value::Number(5.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Subtract,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "1\n");
    }

    #[test]
    fn subtract_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Subtract,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn multiplication() {
        let instructions = vec![
            Op::Constant(Value::Number(5.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Multiply,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "20\n");
    }

    #[test]
    fn multiply_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Multiply,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn division() {
        let instructions = vec![
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Divide,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "5\n");
    }

    #[test]
    fn divide_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::Divide,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn compare_less() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::LessThan,
            Op::Print,
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::LessThan,
            Op::Print,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::LessThan,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "false\nfalse\ntrue\n");
    }

    #[test]
    fn less_than_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::LessThan,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn compare_less_or_equal() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::LessThanOrEqual,
            Op::Print,
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::LessThanOrEqual,
            Op::Print,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::LessThanOrEqual,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "false\ntrue\ntrue\n");
    }

    #[test]
    fn less_than_or_equal_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::LessThanOrEqual,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn compare_greater() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::GreaterThan,
            Op::Print,
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::GreaterThan,
            Op::Print,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::GreaterThan,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "true\nfalse\nfalse\n");
    }

    #[test]
    fn greater_than_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::GreaterThan,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn compare_greater_or_equal() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(2.0)),
            Op::GreaterThanOrEqual,
            Op::Print,
            Op::Constant(Value::Number(20.0)),
            Op::Constant(Value::Number(20.0)),
            Op::GreaterThanOrEqual,
            Op::Print,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(4.0)),
            Op::GreaterThanOrEqual,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "true\ntrue\nfalse\n");
    }

    #[test]
    fn greater_than_or_equal_bool() {
        let instructions = vec![
            Op::Constant(Value::Number(1.3)),
            Op::Constant(Value::True),
            Op::GreaterThanOrEqual,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::Bool))
        );
    }

    #[test]
    fn equals() {
        let instructions = vec![
            Op::Constant(Value::Number(4.0)),
            Op::Constant(Value::Number(4.0)),
            Op::Equals,
            Op::Print,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(20.0)),
            Op::Equals,
            Op::Print,
            Op::Constant(Value::True),
            Op::Constant(Value::True),
            Op::Equals,
            Op::Print,
            Op::Constant(Value::False),
            Op::Constant(Value::False),
            Op::Equals,
            Op::Print,
            Op::Constant(Value::True),
            Op::Constant(Value::False),
            Op::Equals,
            Op::Print,
            Op::Constant("hello".into()),
            Op::Constant("hello".into()),
            Op::Equals,
            Op::Print,
            Op::Constant("hello".into()),
            Op::Constant("world".into()),
            Op::Equals,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
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
            Op::NotEquals,
            Op::Print,
            Op::Constant(Value::Number(2.0)),
            Op::Constant(Value::Number(20.0)),
            Op::NotEquals,
            Op::Print,
            Op::Constant(Value::True),
            Op::Constant(Value::True),
            Op::NotEquals,
            Op::Print,
            Op::Constant(Value::False),
            Op::Constant(Value::False),
            Op::NotEquals,
            Op::Print,
            Op::Constant(Value::True),
            Op::Constant(Value::False),
            Op::NotEquals,
            Op::Print,
            Op::Constant("hello".into()),
            Op::Constant("hello".into()),
            Op::NotEquals,
            Op::Print,
            Op::Constant("hello".into()),
            Op::Constant("world".into()),
            Op::NotEquals,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(
            test_writer.to_string(),
            "false\ntrue\nfalse\nfalse\ntrue\nfalse\ntrue\n"
        );
    }

    #[test]
    fn global_variables_declare_and_access() {
        let instructions = vec![
            Op::Constant(7.0.into()),
            Op::DefineGlobal("x".into()),
            Op::Constant(3.0.into()),
            Op::DefineGlobal("y".into()),
            Op::GetGlobal("x".into()),
            Op::GetGlobal("y".into()),
            Op::Add,
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "10\n");
    }

    #[test]
    fn global_variables_overwrite_declaration() {
        let instructions = vec![
            Op::Constant(7.0.into()),
            Op::DefineGlobal("x".into()),
            Op::Constant(3.0.into()),
            Op::DefineGlobal("x".into()),
            Op::GetGlobal("x".into()),
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "3\n");
    }

    #[test]
    fn assign_to_global() {
        let instructions = vec![
            Op::Constant(7.0.into()),
            Op::DefineGlobal("x".into()),
            Op::Constant(3.0.into()),
            Op::SetGlobal("x".into()),
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "3\n");
    }

    #[test]
    fn assign_to_undefined_global() {
        let instructions = vec![Op::SetGlobal("x".into()), Op::Print];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Undefined("x".into()))
        );
    }

    #[test]
    fn access_undefined_global() {
        let instructions = vec![Op::SetGlobal("x".into()), Op::Print];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Undefined("x".into()))
        );
    }

    #[test]
    fn local_variables_declare_and_access() {
        let instructions = vec![
            Op::Constant(7.0.into()),
            Op::Constant(3.0.into()),
            Op::GetLocal(0),
            Op::GetLocal(1),
            Op::Add,
            Op::Print,
            Op::PopN(2),
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "10\n");
    }

    #[test]
    fn assign_to_local() {
        let instructions = vec![
            Op::Constant(7.0.into()),
            Op::Constant(3.0.into()),
            Op::SetLocal(0),
            Op::GetLocal(0),
            Op::Print,
            Op::PopN(1),
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "3\n");
    }

    #[test]
    fn call_global_void_function_no_args() {
        let instructions = vec![
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "f".into(),
                arity: 0,
                chunk: vec![Op::Constant(1.0.into()), Op::Print, Op::VoidReturn].into(),
            }))),
            Op::DefineGlobal("f".into()),
            Op::GetGlobal("f".into()),
            Op::Call(0),
            Op::Pop,
            Op::GetGlobal("f".into()),
            Op::Call(0),
            Op::Pop,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "1\n1\n");
    }

    #[test]
    fn call_global_void_function_with_args() {
        let instructions = vec![
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "f".into(),
                arity: 2,
                chunk: vec![
                    Op::GetLocal(0),
                    Op::GetLocal(1),
                    Op::Add,
                    Op::Print,
                    Op::VoidReturn,
                ]
                .into(),
            }))),
            Op::DefineGlobal("f".into()),
            // define local variable of 1
            Op::Constant(1.0.into()),
            // access that local
            // note that access of local at 0 also happens in the function defined above - call
            // frames should manage access for each variable in the proper manner
            Op::GetLocal(0),
            Op::Print,
            // define call args here
            Op::Constant(2.0.into()),
            Op::Constant(3.0.into()),
            Op::GetGlobal("f".into()),
            Op::Call(2),
            Op::Pop,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "1\n5\n");
    }

    #[test]
    fn call_global_function_no_args() {
        let instructions = vec![
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "f".into(),
                arity: 0,
                chunk: vec![Op::Constant(1.0.into()), Op::Return].into(),
            }))),
            Op::DefineGlobal("f".into()),
            Op::GetGlobal("f".into()),
            Op::Call(0),
            Op::Print,
            Op::GetGlobal("f".into()),
            Op::Call(0),
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        assert_eq!(test_writer.to_string(), "1\n1\n");
    }

    #[test]
    fn call_function_with_interior_locals() {
        let instructions = vec![
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "f".into(),
                arity: 2,
                chunk: vec![
                    Op::GetLocal(0),
                    Op::GetLocal(1),
                    Op::Add,
                    Op::GetLocal(2),
                    Op::Return,
                ]
                .into(),
            }))),
            Op::DefineGlobal("f".into()),
            Op::Constant(1.0.into()),
            Op::Constant(2.0.into()),
            Op::GetLocal(0),
            Op::GetLocal(1),
            Op::GetGlobal("f".into()),
            Op::Call(2),
            Op::Print,
            Op::GetLocal(0),
            Op::GetLocal(1),
            Op::GetGlobal("f".into()),
            Op::Call(2),
            Op::Print,
            Op::PopN(2),
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        // stack should always be 0 after execution
        assert_eq!(vm.stack.size(), 0);
        assert_eq!(test_writer.to_string(), "3\n3\n");
    }

    #[test]
    fn call_function_mutate_arg() {
        let instructions = vec![
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "f".into(),
                arity: 2,
                chunk: vec![
                    Op::Constant(3.0.into()),
                    Op::SetLocal(0),
                    Op::GetLocal(0),
                    Op::GetLocal(1),
                    Op::Add,
                    Op::Return,
                ]
                .into(),
            }))),
            Op::DefineGlobal("f".into()),
            Op::Constant(1.0.into()),
            Op::Constant(2.0.into()),
            Op::GetLocal(0),
            Op::GetLocal(1),
            Op::GetGlobal("f".into()),
            Op::Call(2),
            Op::Print,
            Op::GetLocal(0),
            Op::Print,
            Op::PopN(2),
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(vm.exec(instructions.into()), Ok(()));
        // stack should always be 0 after execution
        assert_eq!(vm.stack.size(), 0);
        assert_eq!(test_writer.to_string(), "5\n1\n");
    }

    #[test]
    fn provide_call_stack_on_error() {
        let instructions = vec![
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "g".into(),
                arity: 0,
                chunk: vec![
                    Op::Constant("hey".into()),
                    Op::Constant(1.0.into()),
                    Op::Add,
                ]
                .into(),
            }))),
            Op::DefineGlobal("g".into()),
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "f".into(),
                arity: 0,
                chunk: vec![Op::GetGlobal("g".into()), Op::Call(0)].into(),
            }))),
            Op::DefineGlobal("f".into()),
            Op::GetGlobal("f".into()),
            Op::Call(0),
            Op::Pop,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::Type(Type::Number, Type::String))
        );
        assert_eq!(vm.call_stack(), vec!["f", "g"].into());
    }

    #[test]
    fn error_function_call_with_too_few_args() {
        let instructions = vec![
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "f".into(),
                arity: 2,
                chunk: vec![
                    Op::GetLocal(0),
                    Op::GetLocal(1),
                    Op::Add,
                    Op::GetLocal(2),
                    Op::Return,
                ]
                .into(),
            }))),
            Op::DefineGlobal("f".into()),
            Op::Constant(1.0.into()),
            Op::GetGlobal("f".into()),
            Op::Call(1),
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::WrongNumArgs {
                name: "f".into(),
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn error_function_call_with_too_many_args() {
        let instructions = vec![
            Op::Constant(Value::Obj(Obj::Function(Function {
                name: "f".into(),
                arity: 2,
                chunk: vec![
                    Op::GetLocal(0),
                    Op::GetLocal(1),
                    Op::Add,
                    Op::GetLocal(2),
                    Op::Return,
                ]
                .into(),
            }))),
            Op::DefineGlobal("f".into()),
            Op::Constant(1.0.into()),
            Op::Constant(1.0.into()),
            Op::Constant(1.0.into()),
            Op::GetGlobal("f".into()),
            Op::Call(3),
            Op::Print,
        ];

        let mut test_writer = TestWriter::new();
        let mut vm = VirtualMachine::new(&mut test_writer);

        assert_eq!(
            vm.exec(instructions.into()),
            Err(VmError::WrongNumArgs {
                name: "f".into(),
                expected: 2,
                actual: 3,
            })
        );
    }
}
