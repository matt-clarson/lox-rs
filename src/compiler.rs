use std::{
    collections::hash_map::RandomState, error::Error, fmt::Display, hash::BuildHasher, mem,
    num::ParseFloatError, rc::Rc,
};

use crate::{
    ast::{
        Comparison, Declaration, Equality, Expression, Factor, ForInitialiser, Primary, Statement,
        Term, Unary, Var,
    },
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

    fn len(&self) -> usize {
        self.ops.len()
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

    pub fn jump(&mut self, idx: usize) {
        self.idx = idx
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
            Declaration::Var(var) => self.compile_var(chunk, var),
            Declaration::Stmt(stmt) => self.compile_statement(chunk, stmt),
        }
    }

    fn compile_statement(
        &mut self,
        chunk: &mut Chunk,
        statement: &Statement,
    ) -> Result<(), CompileError> {
        match statement {
            Statement::If {
                condition,
                body,
                else_body,
            } => {
                self.compile_expression(chunk, condition)?;
                let jump_if = self.jump_if_false(chunk);
                self.compile_statement(chunk, body)?;
                if let Some(else_body) = else_body {
                    let jump_else = self.jump(chunk);
                    jump_if.patch(chunk);
                    self.compile_statement(chunk, else_body)?;
                    jump_else.patch(chunk);
                } else {
                    jump_if.patch(chunk);
                }
                Ok(())
            }
            Statement::While { condition, body } => {
                let condition_idx = chunk.len();
                self.compile_expression(chunk, condition)?;
                let jump = self.jump_if_false(chunk);
                self.compile_statement(chunk, body)?;
                chunk.push(Op::Jump(condition_idx));
                jump.patch(chunk);
                Ok(())
            }
            Statement::For {
                initialiser,
                condition,
                incrementer,
                body,
            } => {
                self.begin_scope();
                match initialiser {
                    Some(ForInitialiser::Var(var)) => self.compile_var(chunk, var)?,
                    Some(ForInitialiser::Expr(expr)) => {
                        self.compile_expression(chunk, expr)?;
                    }
                    None => (),
                };
                let loop_idx = chunk.len();

                if let Some(expr) = condition {
                    self.compile_expression(chunk, expr)?;
                    let jump = self.jump_if_false(chunk);
                    self.compile_statement(chunk, body)?;

                    if let Some(expr) = incrementer {
                        self.compile_expression(chunk, expr)?;
                        chunk.push(Op::Pop);
                    }

                    chunk.push(Op::Jump(loop_idx));
                    jump.patch(chunk);
                } else {
                    self.compile_statement(chunk, body)?;

                    if let Some(expr) = incrementer {
                        self.compile_expression(chunk, expr)?;
                        chunk.push(Op::Pop);
                    }

                    chunk.push(Op::Jump(loop_idx));
                }
                self.end_scope(chunk);
                Ok(())
            }
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
                self.compile_expression(chunk, expr)?;
                chunk.push(Op::Pop);
                Ok(())
            }
        }
    }

    fn compile_var(&mut self, chunk: &mut Chunk, var: &Var) -> Result<(), CompileError> {
        if self.scope_depth == 0 {
            self.compile_expression(chunk, &var.expr)?;
            let ident = self.span(&var.name);
            chunk.push(Op::DefineGlobal(ident.into()));
        } else {
            let ident = self.span(&var.name);
            let local = self.locals.declare(ident.into(), self.scope_depth)?;
            self.compile_expression(chunk, &var.expr)?;
            local.initialise(&mut self.locals)?;
        }
        Ok(())
    }

    fn compile_expression(&self, chunk: &mut Chunk, expr: &Expression) -> Result<(), CompileError> {
        match expr {
            Expression::Primary(Primary::Number(t)) => {
                let n = self.span(t).parse::<f64>()?;
                let op = Op::Constant(n.into());
                chunk.push(op);
                Ok(())
            }
            Expression::Primary(Primary::True) => {
                let op = Op::Constant(Value::True);
                chunk.push(op);
                Ok(())
            }
            Expression::Primary(Primary::False) => {
                let op = Op::Constant(Value::False);
                chunk.push(op);
                Ok(())
            }
            Expression::Primary(Primary::Nil) => {
                let op = Op::Constant(Value::Nil);
                chunk.push(op);
                Ok(())
            }
            Expression::Primary(Primary::String(t)) => {
                let s = self.span(&t.inner());
                let op = Op::Constant(s.into());
                chunk.push(op);
                Ok(())
            }
            Expression::Primary(Primary::Ident(t)) => {
                let s = self.span(t);
                let op = if let Some(idx) = self.locals.resolve(s)? {
                    Op::GetLocal(idx)
                } else {
                    Op::GetGlobal(s.into())
                };
                chunk.push(op);
                Ok(())
            }
            Expression::Primary(Primary::Group(expr)) => {
                self.compile_expression(chunk, expr)?;
                Ok(())
            }
            Expression::Call { callee, args } => {
                for arg in args.iter() {
                    self.compile_expression(chunk, arg)?;
                }
                self.compile_expression(chunk, callee)?;
                chunk.push(Op::Call(args.len() as u8));
                Ok(())
            }
            Expression::Unary(Unary::Negate(negate)) => {
                self.compile_expression(chunk, negate)?;
                chunk.push(Op::Negate);
                Ok(())
            }
            Expression::Unary(Unary::Not(not)) => {
                self.compile_expression(chunk, not)?;
                chunk.push(Op::Not);
                Ok(())
            }
            Expression::Factor(Factor::Multiply { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Multiply);
                Ok(())
            }
            Expression::Factor(Factor::Divide { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Divide);
                Ok(())
            }
            Expression::Term(Term::Plus { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Add);
                Ok(())
            }
            Expression::Term(Term::Minus { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Subtract);
                Ok(())
            }
            Expression::Comparison(Comparison::LessThan { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::LessThan);
                Ok(())
            }
            Expression::Comparison(Comparison::LessThanOrEquals { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::LessThanOrEqual);
                Ok(())
            }
            Expression::Comparison(Comparison::GreaterThan { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::GreaterThan);
                Ok(())
            }
            Expression::Comparison(Comparison::GreaterThanOrEquals { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::GreaterThanOrEqual);
                Ok(())
            }
            Expression::Equality(Equality::Equals { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::Equals);
                Ok(())
            }
            Expression::Equality(Equality::NotEquals { left, right }) => {
                self.compile_expression(chunk, left)?;
                self.compile_expression(chunk, right)?;
                chunk.push(Op::NotEquals);
                Ok(())
            }
            Expression::Or { left, right } => {
                self.compile_expression(chunk, left)?;
                let jump = self.jump_or(chunk);
                self.compile_expression(chunk, right)?;
                jump.patch(chunk);
                Ok(())
            }
            Expression::And { left, right } => {
                self.compile_expression(chunk, left)?;
                let jump = self.jump_and(chunk);
                self.compile_expression(chunk, right)?;
                jump.patch(chunk);
                Ok(())
            }
            Expression::Assignment { ident, expr } => {
                self.compile_expression(chunk, expr)?;
                let s = self.span(ident);
                if let Some(idx) = self.locals.resolve(s)? {
                    chunk.push(Op::SetLocal(idx));
                    Ok(())
                } else {
                    chunk.push(Op::SetGlobal(s.into()));
                    Ok(())
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

    fn jump_if_false(&self, chunk: &mut Chunk) -> PatchableJump {
        let idx = chunk.ops.len();
        chunk.push(Op::JumpIfFalse(0));
        PatchableJump::IfFalse(idx)
    }

    fn jump(&self, chunk: &mut Chunk) -> PatchableJump {
        let idx = chunk.ops.len();
        chunk.push(Op::Jump(0));
        PatchableJump::Absolute(idx)
    }

    fn jump_or(&self, chunk: &mut Chunk) -> PatchableJump {
        let idx = chunk.ops.len();
        chunk.push(Op::JumpOr(0));
        PatchableJump::Or(idx)
    }

    fn jump_and(&self, chunk: &mut Chunk) -> PatchableJump {
        let idx = chunk.ops.len();
        chunk.push(Op::JumpAnd(0));
        PatchableJump::And(idx)
    }
}

enum PatchableJump {
    Absolute(usize),
    IfFalse(usize),
    Or(usize),
    And(usize),
}

impl PatchableJump {
    fn patch(self, chunk: &mut Chunk) {
        let jump_idx = chunk.ops.len();
        let (idx, jump) = match self {
            Self::Absolute(idx) => (idx, Op::Jump(jump_idx)),
            Self::IfFalse(idx) => (idx, Op::JumpIfFalse(jump_idx)),
            Self::Or(idx) => (idx, Op::JumpOr(jump_idx)),
            Self::And(idx) => (idx, Op::JumpAnd(jump_idx)),
        };
        unsafe {
            let op = chunk.ops.get_unchecked_mut(idx);
            let _ = mem::replace(op, jump);
        }
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
                Op::Pop,
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

    #[test]
    fn compile_if_statement() {
        let s = "
if (false) {
    print \"is false\";
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::False),
                Op::JumpIfFalse(4),
                Op::Constant("is false".into()),
                Op::Print
            ]
            .into())
        );
    }

    #[test]
    fn compile_if_else_statement() {
        let s = "
if (false) {
    print \"is false\";
} else {
    print \"is not false\";
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::False),
                Op::JumpIfFalse(5),
                Op::Constant("is false".into()),
                Op::Print,
                Op::Jump(7),
                Op::Constant("is not false".into()),
                Op::Print
            ]
            .into())
        );
    }

    #[test]
    fn compile_or_expression() {
        let s = "true or false;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::True),
                Op::JumpOr(3),
                Op::Constant(Value::False),
                Op::Pop
            ]
            .into())
        );
    }

    #[test]
    fn compile_and_expression() {
        let s = "true and false;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::True),
                Op::JumpAnd(3),
                Op::Constant(Value::False),
                Op::Pop
            ]
            .into())
        );
    }

    #[test]
    fn compile_while_loop() {
        let s = "
while (true) {
    print \"loop\";
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::True),
                Op::JumpIfFalse(5),
                Op::Constant("loop".into()),
                Op::Print,
                Op::Jump(0),
            ]
            .into())
        );
    }

    #[test]
    fn compile_common_for_loop() {
        let s = "
for (var i=0; i<5; i = i + 1) {
    print i;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(0.0.into()),
                Op::GetLocal(0),
                Op::Constant(5.0.into()),
                Op::LessThan,
                Op::JumpIfFalse(13),
                Op::GetLocal(0),
                Op::Print,
                Op::GetLocal(0),
                Op::Constant(1.0.into()),
                Op::Add,
                Op::SetLocal(0),
                Op::Pop,
                Op::Jump(1),
                Op::PopN(1),
            ]
            .into())
        );
    }

    #[test]
    fn compile_for_loop_expression_initialiser() {
        let s = "
var i;
for (i=0; i<5; i = i + 1) {
    print i;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Nil),
                Op::DefineGlobal("i".into()),
                Op::Constant(0.0.into()),
                Op::SetGlobal("i".into()),
                Op::GetGlobal("i".into()),
                Op::Constant(5.0.into()),
                Op::LessThan,
                Op::JumpIfFalse(16),
                Op::GetGlobal("i".into()),
                Op::Print,
                Op::GetGlobal("i".into()),
                Op::Constant(1.0.into()),
                Op::Add,
                Op::SetGlobal("i".into()),
                Op::Pop,
                Op::Jump(4),
            ]
            .into())
        );
    }

    #[test]
    fn compile_for_loop_no_initialiser() {
        let s = "
var i=0;
for (; true; i = i + 1) {
    print i;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(0.0.into()),
                Op::DefineGlobal("i".into()),
                Op::Constant(Value::True),
                Op::JumpIfFalse(12),
                Op::GetGlobal("i".into()),
                Op::Print,
                Op::GetGlobal("i".into()),
                Op::Constant(1.0.into()),
                Op::Add,
                Op::SetGlobal("i".into()),
                Op::Pop,
                Op::Jump(2),
            ]
            .into())
        );
    }

    #[test]
    fn compile_for_loop_no_condition() {
        let s = "
var i=0;
for (;; i = i + 1) {
    print i;
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(0.0.into()),
                Op::DefineGlobal("i".into()),
                Op::GetGlobal("i".into()),
                Op::Print,
                Op::GetGlobal("i".into()),
                Op::Constant(1.0.into()),
                Op::Add,
                Op::SetGlobal("i".into()),
                Op::Pop,
                Op::Jump(2),
            ]
            .into())
        );
    }

    #[test]
    fn compile_empty_for_loop() {
        let s = "
for (;;) {
    print \"loop\";
}
        "
        .trim();

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant("loop".into()),
                Op::Print,
                Op::Jump(0),
            ]
            .into())
        );
    }
}
