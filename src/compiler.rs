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
        }
    }

    fn compile_statement(
        &self,
        ops: &mut Vec<Op>,
        statement: &Statement,
    ) -> Result<(), CompileError> {
        match statement {
            Statement::Print(print) => {
                self.compile_expression(ops, &print.expr)?;
                ops.push(Op::Print(print.keyword));
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
            Expression::Primary(Primary::True(_)) => {
                let op = Op::Constant(Value::True);
                ops.push(op);
            }
            Expression::Primary(Primary::False(_)) => {
                let op = Op::Constant(Value::False);
                ops.push(op);
            }
            Expression::Primary(Primary::Nil(_)) => {
                let op = Op::Constant(Value::Nil);
                ops.push(op);
            }
            Expression::Primary(Primary::String(t)) => {
                let s = self.span(&t.inner());
                let op = Op::Constant(s.into());
                ops.push(op);
            }
            Expression::Primary(Primary::Group(group)) => {
                self.compile_expression(ops, &group.expr)?;
            }
            Expression::Unary(Unary::Negate(negate)) => {
                self.compile_expression(ops, &negate.expr)?;
                let op = Op::Negate(negate.operator);
                ops.push(op);
            }
            Expression::Unary(Unary::Not(not)) => {
                self.compile_expression(ops, &not.expr)?;
                let op = Op::Not(not.operator);
                ops.push(op);
            }
            Expression::Factor(Factor::Multiply(mult)) => {
                self.compile_expression(ops, &mult.left)?;
                self.compile_expression(ops, &mult.right)?;
                let op = Op::Multiply(mult.operator);
                ops.push(op);
            }
            Expression::Factor(Factor::Divide(div)) => {
                self.compile_expression(ops, &div.left)?;
                self.compile_expression(ops, &div.right)?;
                let op = Op::Divide(div.operator);
                ops.push(op);
            }
            Expression::Term(Term::Plus(plus)) => {
                self.compile_expression(ops, &plus.left)?;
                self.compile_expression(ops, &plus.right)?;
                let op = Op::Add(plus.operator);
                ops.push(op);
            }
            Expression::Term(Term::Minus(minus)) => {
                self.compile_expression(ops, &minus.left)?;
                self.compile_expression(ops, &minus.right)?;
                let op = Op::Subtract(minus.operator);
                ops.push(op);
            }
            Expression::Comparison(Comparison::LessThan(lt)) => {
                self.compile_expression(ops, &lt.left)?;
                self.compile_expression(ops, &lt.right)?;
                let op = Op::LessThan(lt.operator);
                ops.push(op);
            }
            Expression::Comparison(Comparison::LessThanOrEquals(lte)) => {
                self.compile_expression(ops, &lte.left)?;
                self.compile_expression(ops, &lte.right)?;
                let op = Op::LessThanOrEqual(lte.operator);
                ops.push(op);
            }
            Expression::Comparison(Comparison::GreaterThan(gt)) => {
                self.compile_expression(ops, &gt.left)?;
                self.compile_expression(ops, &gt.right)?;
                let op = Op::GreaterThan(gt.operator);
                ops.push(op);
            }
            Expression::Comparison(Comparison::GreaterThanOrEquals(gte)) => {
                self.compile_expression(ops, &gte.left)?;
                self.compile_expression(ops, &gte.right)?;
                let op = Op::GreaterThanOrEqual(gte.operator);
                ops.push(op);
            }
            Expression::Equality(Equality::Equals(eq)) => {
                self.compile_expression(ops, &eq.left)?;
                self.compile_expression(ops, &eq.right)?;
                let op = Op::Equals(eq.operator);
                ops.push(op);
            }
            Expression::Equality(Equality::NotEquals(neq)) => {
                self.compile_expression(ops, &neq.left)?;
                self.compile_expression(ops, &neq.right)?;
                let op = Op::NotEquals(neq.operator);
                ops.push(op);
            }
            _ => unimplemented!(),
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
            Ok(vec![
                Op::Constant(Value::False),
                Op::Not(Span {
                    start: 0,
                    length: 1,
                    line: 1
                }),
                Op::Pop,
                Op::Return
            ]
            .into())
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
                Op::Negate(Span {
                    start: 0,
                    length: 1,
                    line: 1
                }),
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
                Op::Add(Span {
                    start: 4,
                    length: 1,
                    line: 1
                }),
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
                Op::Subtract(Span {
                    start: 4,
                    length: 1,
                    line: 1
                }),
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
                Op::Add(Span {
                    start: 8,
                    length: 1,
                    line: 1
                }),
                Op::Constant(Value::Obj(Obj::String("world".into()))),
                Op::Add(Span {
                    start: 14,
                    length: 1,
                    line: 1
                }),
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
                Op::Multiply(Span {
                    start: 4,
                    length: 1,
                    line: 1
                }),
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
                Op::Divide(Span {
                    start: 4,
                    length: 1,
                    line: 1
                }),
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
                Op::LessThan(Span {
                    start: 4,
                    length: 1,
                    line: 1
                }),
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
                Op::LessThanOrEqual(Span {
                    start: 4,
                    length: 2,
                    line: 1
                }),
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
                Op::GreaterThan(Span {
                    start: 4,
                    length: 1,
                    line: 1
                }),
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
                Op::GreaterThanOrEqual(Span {
                    start: 4,
                    length: 2,
                    line: 1
                }),
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
                Op::Equals(Span {
                    start: 4,
                    length: 2,
                    line: 1
                }),
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
                Op::NotEquals(Span {
                    start: 4,
                    length: 2,
                    line: 1
                }),
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
                Op::Add(Span {
                    start: 7,
                    length: 1,
                    line: 1
                }),
                Op::Multiply(Span {
                    start: 2,
                    length: 1,
                    line: 1
                }),
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
            Ok(vec![
                Op::Constant(Value::Number(123.0)),
                Op::Print(Span {
                    start: 0,
                    length: 5,
                    line: 1
                }),
                Op::Return
            ]
            .into())
        );
    }
}
