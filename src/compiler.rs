use std::{error::Error, fmt::Display, num::ParseFloatError};

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

struct SourceCompiler<'c> {
    source: &'c str,
    parser: Parser<'c>,
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
        let mut c = SourceCompiler { source, parser };

        c.compile()
    }
}

impl<'c> SourceCompiler<'c> {
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
        &self,
        ops: &mut Vec<Op>,
        declaration: &Declaration,
    ) -> Result<(), CompileError> {
        match declaration {
            Declaration::Stmt(stmt) => self.compile_statement(ops, stmt),
            Declaration::Var { name, expr } => {
                self.compile_expression(ops, expr)?;
                let ident = self.span(name);
                ops.push(Op::DefineGlobal(ident.into()));
                Ok(())
            }
        }
    }

    fn compile_statement(
        &self,
        ops: &mut Vec<Op>,
        statement: &Statement,
    ) -> Result<(), CompileError> {
        match statement {
            Statement::Print(expr) => {
                self.compile_expression(ops, expr)?;
                ops.push(Op::Print);
                Ok(())
            }
            Statement::Expr(expr) => {
                self.compile_expression(ops, expr)?;
                ops.push(Op::Pop);
                Ok(())
            }
        }
    }

    fn compile_expression(&self, ops: &mut Vec<Op>, expr: &Expression) -> Result<(), CompileError> {
        match expr {
            Expression::Primary(Primary::Number(t)) => {
                let n = self.span(t).parse::<f64>()?;
                let op = Op::Constant(n.into());
                ops.push(op);
            }
            Expression::Primary(Primary::True) => {
                let op = Op::Constant(Value::True);
                ops.push(op);
            }
            Expression::Primary(Primary::False) => {
                let op = Op::Constant(Value::False);
                ops.push(op);
            }
            Expression::Primary(Primary::Nil) => {
                let op = Op::Constant(Value::Nil);
                ops.push(op);
            }
            Expression::Primary(Primary::String(t)) => {
                let s = self.span(&t.inner());
                let op = Op::Constant(s.into());
                ops.push(op);
            }
            Expression::Primary(Primary::Ident(t)) => {
                let s = self.span(t);
                let op = Op::GetGlobal(s.into());
                ops.push(op);
            }
            Expression::Primary(Primary::Group(expr)) => {
                self.compile_expression(ops, expr)?;
            }
            Expression::Unary(Unary::Negate(negate)) => {
                self.compile_expression(ops, negate)?;
                ops.push(Op::Negate);
            }
            Expression::Unary(Unary::Not(not)) => {
                self.compile_expression(ops, not)?;
                ops.push(Op::Not);
            }
            Expression::Factor(Factor::Multiply { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Multiply);
            }
            Expression::Factor(Factor::Divide { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Divide);
            }
            Expression::Term(Term::Plus { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Add);
            }
            Expression::Term(Term::Minus { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Subtract);
            }
            Expression::Comparison(Comparison::LessThan { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::LessThan);
            }
            Expression::Comparison(Comparison::LessThanOrEquals { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::LessThanOrEqual);
            }
            Expression::Comparison(Comparison::GreaterThan { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::GreaterThan);
            }
            Expression::Comparison(Comparison::GreaterThanOrEquals { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::GreaterThanOrEqual);
            }
            Expression::Equality(Equality::Equals { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::Equals);
            }
            Expression::Equality(Equality::NotEquals { left, right }) => {
                self.compile_expression(ops, left)?;
                self.compile_expression(ops, right)?;
                ops.push(Op::NotEquals);
            }
            Expression::Assignment {ident, expr} => {
                self.compile_expression(ops, expr)?;
                let s = self.span(ident);
                ops.push(Op::SetGlobal(s.into()));
            }
        };
        Ok(())
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
}

impl Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => e.fmt(f),
            Self::InvalidNumber(e) => write!(f, "invalid number: {e}"),
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
            Ok(vec![
                Op::GetGlobal("x".into()),
                Op::Pop,
                Op::Return
            ]
            .into())
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
}
