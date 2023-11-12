use std::{
    collections::hash_map::RandomState, error::Error, fmt::Display, hash::BuildHasher,
    num::ParseFloatError, rc::Rc,
};

use crate::{
    ast::{Comparison, Declaration, Equality, Expression, Factor, Primary, Statement, Term, Unary},
    parser::{ParseError, Parser},
    scanner::{Scanner, Span},
    vm::{Function, Op, Value},
};

/// Compiles source code to a list of bytecode instructions to be consumed by the
/// [VM](crate::vm::VirtualMachine).
///
/// See the [crate docs](crate) for documentation on using the compiler with the VM.
pub struct Compiler {}

struct SourceCompiler<'c, H: BuildHasher> {
    source: &'c str,
    parser: Parser<'c>,
    scope_depth: usize,
    locals: Locals<H>,
}

#[derive(Debug, PartialEq)]
pub struct Chunk {
    ops: Vec<Op>,
}

impl From<Vec<Op>> for Chunk {
    fn from(ops: Vec<Op>) -> Self {
        Self { ops }
    }
}

impl Chunk {
    fn new() -> Self {
        Self { ops: Vec::new() }
    }

    fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    fn into_ref(self) -> Rc<[Op]> {
        self.ops.into()
    }

    pub fn iter(&self) -> ChunkIter<'_> {
        ChunkIter {
            ops: self.ops.as_slice(),
            idx: 0,
        }
    }
}

pub struct ChunkIter<'c> {
    ops: &'c [Op],
    idx: usize,
}

impl<'c> ChunkIter<'c> {
    pub fn new(ops: &'c [Op]) -> Self {
        Self { ops, idx: 0 }
    }
}

impl<'c> Iterator for ChunkIter<'c> {
    type Item = &'c Op;

    fn next(&mut self) -> Option<Self::Item> {
        let op = self.ops.get(self.idx)?;
        self.idx += 1;
        Some(op)
    }
}

struct UninitalisedLocal {
    idx: usize,
    depth: usize,
}

impl UninitalisedLocal {
    fn initialise<H: BuildHasher>(&self, locals: &mut Locals<H>) -> Result<(), CompileError> {
        let local = locals
            .locals
            .get_mut(self.idx)
            .ok_or(CompileError::Initialise)?;
        local.depth = Some(self.depth);
        Ok(())
    }
}

struct Locals<H: BuildHasher> {
    hasher: H,
    locals: Vec<Local>,
}

impl Default for Locals<RandomState> {
    fn default() -> Self {
        let hasher = RandomState::new();
        Self::new(hasher)
    }
}

impl<H: BuildHasher> Locals<H> {
    fn new(hasher: H) -> Self {
        Self {
            hasher,
            locals: vec![],
        }
    }

    /// declares a new local variable - this variable must then be initialised, by calling
    /// [`UninitalisedLocal::initialise`] on the returned object.
    /// These steps are separate to allow for checking for circular definitions of variables.
    fn declare(&mut self, name: Box<str>, depth: usize) -> Result<UninitalisedLocal, CompileError> {
        let hash = self.hasher.hash_one(&name);
        let local = Local {
            hash,
            name,
            depth: None,
        };

        for candidate in self.locals.iter().rev() {
            if candidate.depth < Some(depth) {
                // no more locals at the target depth
                break;
            }
            if candidate.hash == local.hash {
                // there is a collision at the given depth
                // exit and don't insert anything
                return Err(CompileError::RedefineVar(local.name));
            }
        }

        self.locals.push(local);

        Ok(UninitalisedLocal {
            idx: self.locals.len() - 1,
            depth,
        })
    }

    fn define(&mut self, name: Box<str>, depth: usize) -> Result<(), CompileError> {
        let local = self.declare(name, depth)?;
        local.initialise(self)
    }

    fn resolve<S: AsRef<str>>(&self, name: S) -> Result<Option<usize>, CompileError> {
        let hash = self.hasher.hash_one(name.as_ref());
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local.hash == hash {
                if local.depth.is_none() {
                    return Err(CompileError::CircularDef(name.as_ref().into()));
                }
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    fn remove_depth(&mut self, depth: usize) -> usize {
        let mut n = 0usize;
        let depth = Some(depth);

        for local in self.locals.iter().rev() {
            if depth == local.depth {
                n += 1;
            } else {
                break;
            }
        }

        self.locals.truncate(self.locals.len() - n);

        n
    }
}

#[derive(Eq, Hash, PartialEq)]
struct Local {
    hash: u64,
    name: Box<str>,
    depth: Option<usize>,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Self {}
    }

    /// Produces bytecode instructions for the given input string.
    pub fn compile(&self, source: &str) -> Result<Chunk, CompileError> {
        let scanner = Scanner::from(source);
        let parser = Parser::from(scanner);
        let mut c = SourceCompiler {
            source,
            parser,
            scope_depth: 0,
            locals: Locals::default(),
        };

        c.compile()
    }
}

impl<'c, H: BuildHasher> SourceCompiler<'c, H> {
    fn compile(&mut self) -> Result<Chunk, CompileError> {
        let mut chunk = Chunk::new();

        while let Some(result) = self.parser.next() {
            let declaration = result?;
            self.compile_declaration(&mut chunk, &declaration)?;
        }

        Ok(chunk)
    }

    fn compile_declaration(
        &mut self,
        chunk: &mut Chunk,
        declaration: &Declaration,
    ) -> Result<(), CompileError> {
        match declaration {
            Declaration::Function { name, args, body } => {
                // function args are defined in an outer scope
                self.begin_function_scope();
                for arg in args.iter() {
                    let ident = self.span(arg);
                    self.locals.define(ident.into(), self.scope_depth)?;
                }
                // function body is defined in an inner scope
                self.begin_function_scope();
                let mut fun_chunk = Chunk::new();
                for decl in body.iter() {
                    self.compile_declaration(&mut fun_chunk, decl)?;
                }

                // end inner scope
                self.end_function_scope();

                let ident: Rc<str> = self.span(name).into();
                let fun = Function {
                    name: Rc::clone(&ident),
                    arity: args.len() as u8,
                    chunk: fun_chunk.into_ref(),
                };
                chunk.push(Op::Constant(fun.into()));

                // end outer scope
                self.end_function_scope();

                if self.scope_depth == 0 {
                    chunk.push(Op::DefineGlobal(ident.into()));
                } else {
                    self.locals
                        .define(ident.as_ref().into(), self.scope_depth)?;
                }
                Ok(())
            }
            Declaration::Stmt(stmt) => self.compile_statement(chunk, stmt),
            Declaration::Var { name, expr } => {
                if self.scope_depth == 0 {
                    self.compile_expression(chunk, expr)?;
                    let ident = self.span(name);
                    chunk.push(Op::DefineGlobal(ident.into()));
                } else {
                    let ident = self.span(name);
                    let local = self.locals.declare(ident.into(), self.scope_depth)?;
                    self.compile_expression(chunk, expr)?;
                    local.initialise(&mut self.locals)?;
                }
                Ok(())
            }
        }
    }

    fn compile_statement(
        &mut self,
        chunk: &mut Chunk,
        statement: &Statement,
    ) -> Result<(), CompileError> {
        match statement {
            Statement::Return(Some(expr)) => {
                self.compile_expression(chunk, expr)?;
                chunk.push(Op::Return);
                Ok(())
            }
            Statement::Return(None) => {
                chunk.push(Op::VoidReturn);
                Ok(())
            }
            Statement::Print(expr) => {
                self.compile_expression(chunk, expr)?;
                chunk.push(Op::Print);
                Ok(())
            }
            Statement::Block(declarations) => {
                self.begin_scope();
                for declaration in declarations.iter() {
                    self.compile_declaration(chunk, declaration)?;
                }
                self.end_scope(chunk);
                Ok(())
            }
            Statement::Expr(expr) => {
                let pop_last = self.compile_expression(chunk, expr)?;
                if pop_last {
                    chunk.push(Op::Pop);
                }
                Ok(())
            }
        }
    }

    fn compile_expression(
        &self,
        chunk: &mut Chunk,
        expr: &Expression,
    ) -> Result<bool, CompileError> {
        match expr {
            Expression::Primary(Primary::Number(t)) => {
                let n = self.span(t).parse::<f64>()?;
                let op = Op::Constant(n.into());
                chunk.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::True) => {
                let op = Op::Constant(Value::True);
                chunk.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::False) => {
                let op = Op::Constant(Value::False);
                chunk.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::Nil) => {
                let op = Op::Constant(Value::Nil);
                chunk.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::String(t)) => {
                let s = self.span(&t.inner());
                let op = Op::Constant(s.into());
                chunk.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::Ident(t)) => {
                let s = self.span(t);
                let op = if let Some(idx) = self.locals.resolve(s)? {
                    Op::GetLocal(idx)
                } else {
                    Op::GetGlobal(s.into())
                };
                chunk.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::Group(expr)) => {
                self.compile_expression(chunk, expr)?;
                Ok(true)
            }
            Expression::Call { callee, args } => {
                for arg in args.iter() {
                    self.compile_expression(chunk, arg)?;
                }
                self.compile_expression(chunk, callee)?;
                chunk.push(Op::Call(args.len() as u8));
                Ok(true)
            }
            Expression::Unary(Unary::Negate(negate)) => {
                self.compile_expression(chunk, negate)?;
                chunk.push(Op::Negate);
                Ok(true)
            }
            Expression::Unary(Unary::Not(not)) => {
                self.compile_expression(chunk, not)?;
                chunk.push(Op::Not);
                Ok(true)
            }
            Expression::Factor(Factor::Multiply { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Multiply);
                Ok(true)
            }
            Expression::Factor(Factor::Divide { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Divide);
                Ok(true)
            }
            Expression::Term(Term::Plus { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Add);
                Ok(true)
            }
            Expression::Term(Term::Minus { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Subtract);
                Ok(true)
            }
            Expression::Comparison(Comparison::LessThan { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::LessThan);
                Ok(true)
            }
            Expression::Comparison(Comparison::LessThanOrEquals { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::LessThanOrEqual);
                Ok(true)
            }
            Expression::Comparison(Comparison::GreaterThan { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::GreaterThan);
                Ok(true)
            }
            Expression::Comparison(Comparison::GreaterThanOrEquals { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::GreaterThanOrEqual);
                Ok(true)
            }
            Expression::Equality(Equality::Equals { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Equals);
                Ok(true)
            }
            Expression::Equality(Equality::NotEquals { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::NotEquals);
                Ok(true)
            }
            Expression::Assignment { ident, expr } => {
                self.compile_expression(chunk, expr)?;
                let s = self.span(ident);
                if let Some(idx) = self.locals.resolve(s)? {
                    chunk.push(Op::SetLocal(idx));
                    Ok(false)
                } else {
                    chunk.push(Op::SetGlobal(s.into()));
                    Ok(true)
                }
            }
        }
    }

    fn begin_scope(&mut self) {
        self.scope_depth += 1;
    }

    fn begin_function_scope(&mut self) {
        self.scope_depth += 1;
    }

    fn end_scope(&mut self, chunk: &mut Chunk) {
        let n = self.locals.remove_depth(self.scope_depth);
        if n > 0 {
            chunk.push(Op::PopN(n));
        }
        self.scope_depth -= 1;
    }

    fn end_function_scope(&mut self) {
        self.locals.remove_depth(self.scope_depth);
        self.scope_depth -= 1;
    }

    fn span(&self, t: &Span) -> &str {
        unsafe { self.source.get_unchecked(t.start..t.start + t.length) }
    }
}

/// Represents possible error states that can be encountered during compilation.
#[derive(Debug, PartialEq)]
pub enum CompileError {
    /// Errors Raised during scanning/parsing.
    Parse(ParseError),
    /// Raised when parsing number literals into concrete values.
    InvalidNumber(ParseFloatError),
    RedefineVar(Box<str>),
    CircularDef(Box<str>),
    Initialise,
}

impl Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => e.fmt(f),
            Self::InvalidNumber(e) => write!(f, "invalid number: {e}"),
            Self::RedefineVar(s) => write!(f, "cannot redefine variable '{s}'"),
            Self::CircularDef(s) => write!(
                f,
                "cannot read variable in its own initialiser: variable '{s}'"
            ),
            Self::Initialise => f.write_str("error initialising local variable"),
        }
    }
}

impl Error for CompileError {}

impl From<ParseError> for CompileError {
    fn from(value: ParseError) -> Self {
        Self::Parse(value)
    }
}

impl From<ParseFloatError> for CompileError {
    fn from(value: ParseFloatError) -> Self {
        Self::InvalidNumber(value)
    }
}

#[cfg(test)]
mod test {
    use crate::vm::{Function, Obj};

    use super::*;

    #[test]
    fn compile_number() {
        let s = "1.4;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::Number(1.4)), Op::Pop].into())
        );
    }

    #[test]
    fn compile_true() {
        let s = "true;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::True), Op::Pop].into())
        );
    }

    #[test]
    fn compile_false() {
        let s = "false;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::False), Op::Pop].into())
        );
    }

    #[test]
    fn compile_nil() {
        let s = "nil;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::Nil), Op::Pop].into())
        );
    }

    #[test]
    fn compile_string() {
        let s = "\"hello\";";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Obj(Obj::String("hello".into()))),
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_not() {
        let s = "!false;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::False), Op::Not, Op::Pop].into())
        );
    }

    #[test]
    fn compile_negate() {
        let s = "-1.4;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::Number(1.4)), Op::Negate, Op::Pop,].into())
        );
    }

    #[test]
    fn compile_addition() {
        let s = "1.4 + 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::Add,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_subtraction() {
        let s = "1.4 - 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::Subtract,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_string_concatenation() {
        let s = "\"hello\" + \" \" + \"world\";";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Obj(Obj::String("hello".into()))),
                Op::Constant(Value::Obj(Obj::String(" ".into()))),
                Op::Add,
                Op::Constant(Value::Obj(Obj::String("world".into()))),
                Op::Add,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_multiplication() {
        let s = "1.4 * 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::Multiply,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_division() {
        let s = "1.4 / 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::Divide,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_less() {
        let s = "1.4 < 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::LessThan,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_less_or_equal() {
        let s = "1.4 <= 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::LessThanOrEqual,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_greater() {
        let s = "1.4 > 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::GreaterThan,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_greater_or_equal() {
        let s = "1.4 >= 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::GreaterThanOrEqual,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_equals() {
        let s = "1.4 == 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::Equals,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_not_equals() {
        let s = "1.4 != 6.5;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Constant(Value::Number(6.5)),
                Op::NotEquals,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_group() {
        let s = "2 * (1 + 5);";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(2.0)),
                Op::Constant(Value::Number(1.0)),
                Op::Constant(Value::Number(5.0)),
                Op::Add,
                Op::Multiply,
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_print() {
        let s = "print 123;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::Number(123.0)), Op::Print].into())
        );
    }

    #[test]
    fn compile_global_declaration() {
        let s = "var x = 7;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(7.0)),
                Op::DefineGlobal("x".into()),
            ]
            .into())
        );
    }

    #[test]
    fn compile_global_declaration_unassigned() {
        let s = "var x;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::Nil), Op::DefineGlobal("x".into()),].into())
        );
    }

    #[test]
    fn compile_global_access() {
        let s = "x;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::GetGlobal("x".into()), Op::Pop].into())
        );
    }

    #[test]
    fn compile_global_assignment() {
        let s = "x = 7;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(7.0.into()), Op::SetGlobal("x".into()), Op::Pop,].into())
        );
    }

    #[test]
    fn compile_block() {
        let s = "
{
    print 1;
    print 2;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(1.0.into()),
                Op::Print,
                Op::Constant(2.0.into()),
                Op::Print,
            ]
            .into())
        );
    }

    #[test]
    fn compile_empty_block() {
        let s = "{}";

        let compiler = Compiler::default();

        assert_eq!(compiler.compile(s), Ok(vec![].into()));
    }

    #[test]
    fn compile_locals_depth_1() {
        let s = "
{
    var x = 1;
    var y = 2;
    print x + y;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(1.0.into()),
                Op::Constant(2.0.into()),
                Op::GetLocal(0),
                Op::GetLocal(1),
                Op::Add,
                Op::Print,
                Op::PopN(2),
            ]
            .into())
        );
    }

    #[test]
    fn compile_locals_collision() {
        let s = "
{
    var x = 1;
    var x = 2;
    print x + y;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Err(CompileError::RedefineVar("x".into()))
        );
    }

    #[test]
    fn compile_locals_shadowing() {
        let s = "
{
    var x = 1;
    {
        var x = 2;
        print x;
    }
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(1.0.into()),
                Op::Constant(2.0.into()),
                Op::GetLocal(1),
                Op::Print,
                Op::PopN(1),
                Op::PopN(1),
            ]
            .into())
        );
    }

    #[test]
    fn compile_locals_circular_def() {
        let s = "
{
    var x = 1;
    {
        var x = x;
        print x;
    }
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Err(CompileError::CircularDef("x".into()))
        );
    }

    #[test]
    fn compile_local_assignment() {
        let s = "
{
    var x = 1;
    x = 2;
    print x;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(1.0.into()),
                Op::Constant(2.0.into()),
                Op::SetLocal(0),
                Op::GetLocal(0),
                Op::Print,
                Op::PopN(1),
            ]
            .into())
        );
    }

    #[test]
    fn compile_void_function_no_args_no_locals() {
        let s = "
fun f() {
    print 123;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Obj(Obj::Function(Function {
                    name: "f".into(),
                    arity: 0,
                    chunk: vec![Op::Constant(123.0.into()), Op::Print, Op::VoidReturn].into()
                }))),
                Op::DefineGlobal("f".into()),
            ]
            .into())
        );
    }

    #[test]
    fn compile_returning_function_no_args_no_locals() {
        let s = "
fun f() {
    return 123;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Obj(Obj::Function(Function {
                    name: "f".into(),
                    arity: 0,
                    chunk: vec![Op::Constant(123.0.into()), Op::Return].into()
                }))),
                Op::DefineGlobal("f".into()),
            ]
            .into())
        );
    }

    #[test]
    fn compile_returning_empty_function_no_args_no_locals() {
        let s = "
fun f() {
    return;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Obj(Obj::Function(Function {
                    name: "f".into(),
                    arity: 0,
                    chunk: vec![Op::VoidReturn].into()
                }))),
                Op::DefineGlobal("f".into()),
            ]
            .into())
        );
    }

    #[test]
    fn compile_void_function_one_arg_no_locals() {
        let s = "
fun f(a) {
    print a;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Obj(Obj::Function(Function {
                    name: "f".into(),
                    arity: 1,
                    chunk: vec![Op::GetLocal(0), Op::Print, Op::VoidReturn].into()
                }))),
                Op::DefineGlobal("f".into()),
            ]
            .into())
        );
    }

    #[test]
    fn compile_void_function_two_args_no_locals() {
        let s = "
fun f(a, b) {
    print a + b;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Obj(Obj::Function(Function {
                    name: "f".into(),
                    arity: 2,
                    chunk: vec![
                        Op::GetLocal(0),
                        Op::GetLocal(1),
                        Op::Add,
                        Op::Print,
                        Op::VoidReturn
                    ]
                    .into()
                }))),
                Op::DefineGlobal("f".into()),
            ]
            .into())
        );
    }

    #[test]
    fn compile_function_call_no_args() {
        let s = "f();";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::GetGlobal("f".into()), Op::Call(0), Op::Pop,].into())
        );
    }

    #[test]
    fn compile_function_call_one_arg() {
        let s = "f(1);";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(1.0.into()),
                Op::GetGlobal("f".into()),
                Op::Call(1),
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_function_call_two_args() {
        let s = "f(1, 2);";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(1.0.into()),
                Op::Constant(2.0.into()),
                Op::GetGlobal("f".into()),
                Op::Call(2),
                Op::Pop,
            ]
            .into())
        );
    }

    #[test]
    fn compile_multi_function_call() {
        let s = "f()();";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::GetGlobal("f".into()), Op::Call(0), Op::Call(0), Op::Pop,].into())
        );
    }

    #[test]
    fn compile_local_defined_function_call() {
        let s = "
{
    fun f(a, b) {
        print a + b;
    }

    f(1, 2);
    f(3, 4);
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
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
                    .into()
                }))),
                Op::Constant(1.0.into()),
                Op::Constant(2.0.into()),
                Op::GetLocal(0),
                Op::Call(2),
                Op::Pop,
                Op::Constant(3.0.into()),
                Op::Constant(4.0.into()),
                Op::GetLocal(0),
                Op::Call(2),
                Op::Pop,
                Op::PopN(1),
            ]
            .into())
        );
    }

    #[test]
    fn compile_function_with_locals() {
        let s = "
fun f(a, b) {
    var s = a + b;
    return s;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Obj(Obj::Function(Function {
                    name: "f".into(),
                    arity: 2,
                    chunk: vec![
                        Op::GetLocal(0),
                        Op::GetLocal(1),
                        Op::Add,
                        Op::GetLocal(2),
                        Op::Return
                    ]
                    .into()
                }))),
                Op::DefineGlobal("f".into())
            ]
            .into())
        );
    }
}
