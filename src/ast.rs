use crate::scanner::Span;

/// A complete statement of the language - this is the highest type of node in the AST.
#[derive(Debug, Eq, PartialEq)]
pub enum Declaration {
    Function {name: Span, args: Box<[Span]>, body: Box<[Declaration]>},
    Var { name: Span, expr: Expression },
    Stmt(Statement),
}

#[derive(Debug, Eq, PartialEq)]
pub enum Statement {
    Return(Option<Expression>),
    Print(Expression),
    Block(Box<[Declaration]>),
    Expr(Expression),
}

impl From<Statement> for Declaration {
    fn from(value: Statement) -> Self {
        Self::Stmt(value)
    }
}

impl From<Expression> for Declaration {
    fn from(value: Expression) -> Self {
        Self::Stmt(Statement::Expr(value))
    }
}

/// A single expression.
#[derive(Debug, Eq, PartialEq)]
pub enum Expression {
    Assignment { ident: Span, expr: Box<Expression> },
    Equality(Equality),
    Comparison(Comparison),
    Term(Term),
    Factor(Factor),
    Unary(Unary),
    Call{
        callee: Box<Expression>,
        args: Box<[Expression]>,
    },
    Primary(Primary),
}

/// An equality expression (e.g. x == y)
#[derive(Debug, Eq, PartialEq)]
pub enum Equality {
    Equals {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    NotEquals {
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

/// A comparison expression (e.g. x < y)
#[derive(Debug, Eq, PartialEq)]
pub enum Comparison {
    LessThan {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    LessThanOrEquals {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    GreaterThan {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    GreaterThanOrEquals {
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

/// A term expression (e.g. x + y)
#[derive(Debug, Eq, PartialEq)]
pub enum Term {
    Plus {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Minus {
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

/// A factor expression (e.g. x / y)
#[derive(Debug, Eq, PartialEq)]
pub enum Factor {
    Multiply {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Divide {
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

/// A unary expression (e.g. !y)
#[derive(Debug, Eq, PartialEq)]
pub enum Unary {
    Negate(Box<Expression>),
    Not(Box<Expression>),
}

/// The smallest expression unit, literals, identifiers, and grouped expressions.
///
/// Groups are included in `Primary` to ensure correct precedence.
#[derive(Debug, Eq, PartialEq)]
pub enum Primary {
    String(Span),
    Number(Span),
    Ident(Span),
    Nil,
    True,
    False,
    Group(Box<Expression>),
}
