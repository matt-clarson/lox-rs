use std::{
    collections::hash_map::RandomState, error::Error, fmt::Display, hash::BuildHasher,
    num::ParseFloatError,
};

use crate::{
    ast::{Comparison, Declaration, Equality, Expression, Factor, Primary, Statement, Term, Unary},
    parser::{ParseError, Parser},
    scanner::{Scanner, Span},
    vm::{Op, Value},
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

struct UninitalisedLocal {
    idx: usize,
    depth: usize,
}

impl UninitalisedLocal {
    fn initialise<H: BuildHasher>(& self, locals: &mut Locals<H>) -> Result<(), CompileError> {
        let local = locals.locals.get_mut(self.idx).ok_or(CompileError::Initialise)?;
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
        let depth = Some(depth);

        for candidate in self.locals.iter().rev() {
            if candidate.depth < depth {
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

        Ok(UninitalisedLocal { idx: self.locals.len()-1, depth: depth.unwrap() })
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
    pub fn compile(&self, source: &str) -> Result<Vec<Op>, CompileError> {
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
    fn compile(&mut self) -> Result<Vec<Op>, CompileError> {
        let mut ops = vec![];

        while let Some(result) = self.parser.next() {
            let declaration = result?;
            self.compile_declaration(&mut ops, &declaration)?;
        }

        // TODO remove this when we get functions
        ops.push(Op::Return);

        Ok(ops)
    }

    fn compile_declaration(
        &mut self,
        ops: &mut Vec<Op>,
        declaration: &Declaration,
    ) -> Result<(), CompileError> {
        match declaration {
            Declaration::Stmt(stmt) => self.compile_statement(ops, stmt),
            Declaration::Var { name, expr } => {
                if self.scope_depth == 0 {
                    self.compile_expression(ops, expr)?;
                    let ident = self.span(name);
                    ops.push(Op::DefineGlobal(ident.into()));
                } else {
                    let ident = self.span(name);
                    let local = self.locals.declare(ident.into(), self.scope_depth)?;
                    self.compile_expression(ops, expr)?;
                    local.initialise(&mut self.locals)?;
                }
                Ok(())
            }
        }
    }

    fn compile_statement(
        &mut self,
        ops: &mut Vec<Op>,
        statement: &Statement,
    ) -> Result<(), CompileError> {
        match statement {
            Statement::Print(expr) => {
                self.compile_expression(ops, expr)?;
                ops.push(Op::Print);
                Ok(())
            }
            Statement::Block(declarations) => {
                self.scope_depth += 1;
                for declaration in declarations.iter() {
                    self.compile_declaration(ops, declaration)?;
                }
                let n = self.locals.remove_depth(self.scope_depth);
                if n > 0 {
                    ops.push(Op::PopN(n));
                }
                self.scope_depth -= 1;
                Ok(())
            }
            Statement::Expr(expr) => {
                let pop_last = self.compile_expression(ops, expr)?;
                if pop_last {
                    ops.push(Op::Pop);
                }
                Ok(())
            }
        }
    }

    fn compile_expression(
        &self,
        ops: &mut Vec<Op>,
        expr: &Expression,
    ) -> Result<bool, CompileError> {
        match expr {
            Expression::Primary(Primary::Number(t)) => {
                let n = self.span(t).parse::<f64>()?;
                let op = Op::Constant(n.into());
                ops.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::True) => {
                let op = Op::Constant(Value::True);
                ops.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::False) => {
                let op = Op::Constant(Value::False);
                ops.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::Nil) => {
                let op = Op::Constant(Value::Nil);
                ops.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::String(t)) => {
                let s = self.span(&t.inner());
                let op = Op::Constant(s.into());
                ops.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::Ident(t)) => {
                let s = self.span(t);
                let op = if let Some(idx) = self.locals.resolve(s)? {
                    Op::GetLocal(idx)
                } else {
                    Op::GetGlobal(s.into())
                };
                ops.push(op);
                Ok(true)
            }
            Expression::Primary(Primary::Group(expr)) => {
                self.compile_expression(ops, expr)?;
                Ok(true)
            }
            Expression::Unary(Unary::Negate(negate)) => {
                self.compile_expression(ops, negate)?;
                ops.push(Op::Negate);
                Ok(true)
            }
            Expression::Unary(Unary::Not(not)) => {
                self.compile_expression(ops, not)?;
                ops.push(Op::Not);
                Ok(true)
            }
            Expression::Factor(Factor::Multiply { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Multiply);
                Ok(true)
            }
            Expression::Factor(Factor::Divide { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Divide);
                Ok(true)
            }
            Expression::Term(Term::Plus { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Add);
                Ok(true)
            }
            Expression::Term(Term::Minus { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Subtract);
                Ok(true)
            }
            Expression::Comparison(Comparison::LessThan { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::LessThan);
                Ok(true)
            }
            Expression::Comparison(Comparison::LessThanOrEquals { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::LessThanOrEqual);
                Ok(true)
            }
            Expression::Comparison(Comparison::GreaterThan { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::GreaterThan);
                Ok(true)
            }
            Expression::Comparison(Comparison::GreaterThanOrEquals { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::GreaterThanOrEqual);
                Ok(true)
            }
            Expression::Equality(Equality::Equals { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Equals);
                Ok(true)
            }
            Expression::Equality(Equality::NotEquals { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::NotEquals);
                Ok(true)
            }
            Expression::Assignment { ident, expr } => {
                self.compile_expression(ops, expr)?;
                let s = self.span(ident);
                if let Some(idx) = self.locals.resolve(s)? {
                    ops.push(Op::SetLocal(idx));
                    Ok(false)
                } else {
                    ops.push(Op::SetGlobal(s.into()));
                    Ok(true)
                }
            }
        }
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
    Initialise
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
            Self::Initialise => f.write_str("error initialising local variable")
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
    use crate::vm::Obj;

    use super::*;

    #[test]
    fn compile_number() {
        let s = "1.4;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::Number(1.4)), Op::Pop, Op::Return].into())
        );
    }

    #[test]
    fn compile_true() {
        let s = "true;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::True), Op::Pop, Op::Return].into())
        );
    }

    #[test]
    fn compile_false() {
        let s = "false;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::False), Op::Pop, Op::Return].into())
        );
    }

    #[test]
    fn compile_nil() {
        let s = "nil;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::Constant(Value::Nil), Op::Pop, Op::Return].into())
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
                Op::Return
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
            Ok(vec![Op::Constant(Value::False), Op::Not, Op::Pop, Op::Return].into())
        );
    }

    #[test]
    fn compile_negate() {
        let s = "-1.4;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(Value::Number(1.4)),
                Op::Negate,
                Op::Pop,
                Op::Return
            ]
            .into())
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
                Op::Return
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
            Ok(vec![Op::Constant(Value::Number(123.0)), Op::Print, Op::Return].into())
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
                Op::Return
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
            Ok(vec![
                Op::Constant(Value::Nil),
                Op::DefineGlobal("x".into()),
                Op::Return
            ]
            .into())
        );
    }

    #[test]
    fn compile_global_access() {
        let s = "x;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![Op::GetGlobal("x".into()), Op::Pop, Op::Return].into())
        );
    }

    #[test]
    fn compile_global_assignment() {
        let s = "x = 7;";

        let compiler = Compiler::default();

        assert_eq!(
            compiler.compile(s),
            Ok(vec![
                Op::Constant(7.0.into()),
                Op::SetGlobal("x".into()),
                Op::Pop,
                Op::Return
            ]
            .into())
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
                Op::Return
            ]
            .into())
        );
    }

    #[test]
    fn compile_empty_block() {
        let s = "{}";

        let compiler = Compiler::default();

        assert_eq!(compiler.compile(s), Ok(vec![Op::Return].into()));
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
                Op::Return
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
                Op::Return
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
                Op::Return
            ]
            .into())
        );
    }
}
