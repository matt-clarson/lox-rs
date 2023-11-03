use crate::scanner::Span;

/// A complete statement of the language - this is the highest type of node in the AST.
#[derive(Debug, Eq, PartialEq)]
pub enum Statement {
    Expr(Expression),
}

impl From<Expression> for Statement {
    fn from(value: Expression) -> Self {
        Self::Expr(value)
    }
}

/// A single expression.
#[derive(Debug, Eq, PartialEq)]
pub enum Expression {
    Equality(Equality),
    Comparison(Comparison),
    Term(Term),
    Factor(Factor),
    Unary(Unary),
    Primary(Primary),
}

/// An equality expression (e.g. x == y)
#[derive(Debug, Eq, PartialEq)]
pub enum Equality {
    Equals(BinaryExpression),
    NotEquals(BinaryExpression),
}

/// A comparison expression (e.g. x < y)
#[derive(Debug, Eq, PartialEq)]
pub enum Comparison {
    LessThan(BinaryExpression),
    LessThanOrEquals(BinaryExpression),
    GreaterThan(BinaryExpression),
    GreaterThanOrEquals(BinaryExpression),
}

/// A term expression (e.g. x + y)
#[derive(Debug, Eq, PartialEq)]
pub enum Term {
    Plus(BinaryExpression),
    Minus(BinaryExpression),
}

/// A factor expression (e.g. x / y)
#[derive(Debug, Eq, PartialEq)]
pub enum Factor {
    Multiply(BinaryExpression),
    Divide(BinaryExpression),
}

/// A unary expression (e.g. !y)
#[derive(Debug, Eq, PartialEq)]
pub enum Unary {
    Negate(UnaryExpression),
    Not(UnaryExpression),
}

/// Contains info needed to represent a generic binary expression.
#[derive(Debug, Eq, PartialEq)]
pub struct BinaryExpression {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub operator: Span,
}

/// Contains info needed to represent a generic unary expression.
#[derive(Debug, Eq, PartialEq)]
pub struct UnaryExpression {
    pub expr: Box<Expression>,
    pub operator: Span,
}

/// The smallest expression unit, literals, identifiers, and grouped expressions.
///
/// Groups are included in `Primary` to ensure correct precedence.
#[derive(Debug, Eq, PartialEq)]
pub enum Primary {
    String(Span),
    Number(Span),
    Ident(Span),
    Nil(Span),
    True(Span),
    False(Span),
    Group(GroupedExpression),
}

/// Contains info needed to represent a grouped expression.
#[derive(Debug, Eq, PartialEq)]
pub struct GroupedExpression {
    pub start: Span,
    pub end: Span,
    pub expr: Box<Expression>,
}
