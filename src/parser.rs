use std::{error::Error, fmt::Display, iter::Peekable};

use crate::{
    ast::*,
    scanner::{ScanError, Scanner, Token, Span},
};

/// Uses a source of scanned [tokens](crate::scanner::Token) to output an AST, as a stream of
/// [statements](crate::ast::Statement).
pub struct Parser<'s> {
    scanner: Peekable<Scanner<'s>>,
}

impl<'s> From<Scanner<'s>> for Parser<'s> {
    fn from(scanner: Scanner<'s>) -> Self {
        Self {
            scanner: scanner.peekable(),
        }
    }
}

impl<'s> Iterator for Parser<'s> {
    type Item = Result<Statement, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished() {
            None
        } else {
            Some(self.statement())
        }
    }
}

type ParseResult<T> = Result<T, ParseError>;

impl<'s> Parser<'s> {
    fn statement(&mut self) -> ParseResult<Statement> {
        self.expression().map(Statement::Expr)
    }

    fn expression(&mut self) -> ParseResult<Expression> {
        self.equality()
    }

    fn equality(&mut self) -> ParseResult<Expression> {
        let mut expr = self.comparison()?;

        loop {
            expr = match self.peek_next() {
                Ok(&Token::EqualEqual(operator)) => {
                    Expression::Equality(Equality::Equals(BinaryExpression {
                        operator,
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.comparison())?),
                    }))
                }
                Ok(&Token::BangEqual(operator)) => {
                    Expression::Equality(Equality::NotEquals(BinaryExpression {
                        operator,
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.comparison())?),
                    }))
                }
                _ => break,
            };
        }

        Ok(expr)
    }

    fn comparison(&mut self) -> ParseResult<Expression> {
        let mut expr = self.term()?;

        loop {
            expr = match self.peek_next() {
                Ok(&Token::Less(operator)) => {
                    Expression::Comparison(Comparison::LessThan(BinaryExpression {
                        operator,
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.term())?),
                    }))
                }
                Ok(&Token::LessEqual(operator)) => {
                    Expression::Comparison(Comparison::LessThanOrEquals(BinaryExpression {
                        operator,
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.term())?),
                    }))
                }
                Ok(&Token::Greater(operator)) => {
                    Expression::Comparison(Comparison::GreaterThan(BinaryExpression {
                        operator,
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.term())?),
                    }))
                }
                Ok(&Token::GreaterEqual(operator)) => {
                    Expression::Comparison(Comparison::GreaterThanOrEquals(BinaryExpression {
                        operator,
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.term())?),
                    }))
                }
                _ => break,
            };
        }

        Ok(expr)
    }

    fn term(&mut self) -> ParseResult<Expression> {
        let mut expr = self.factor()?;

        loop {
            expr = match self.peek_next() {
                Ok(&Token::Minus(operator)) => Expression::Term(Term::Minus(BinaryExpression {
                    operator,
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.factor())?),
                })),
                Ok(&Token::Plus(operator)) => Expression::Term(Term::Plus(BinaryExpression {
                    operator,
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.factor())?),
                })),
                _ => break,
            };
        }

        Ok(expr)
    }

    fn factor(&mut self) -> ParseResult<Expression> {
        let mut expr = self.unary()?;

        loop {
            expr = match self.peek_next() {
                Ok(&Token::Slash(operator)) => {
                    Expression::Factor(Factor::Divide(BinaryExpression {
                        operator,
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.unary())?),
                    }))
                }
                Ok(&Token::Star(operator)) => {
                    Expression::Factor(Factor::Multiply(BinaryExpression {
                        operator,
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.unary())?),
                    }))
                }
                _ => break,
            };
        }

        Ok(expr)
    }

    fn unary(&mut self) -> ParseResult<Expression> {
        match self.peek_next()? {
            Token::Minus(operator) => Ok(Expression::Unary(Unary::Negate(UnaryExpression {
                operator: *operator,
                expr: Box::new(self.advance().and_then(|_| self.primary())?),
            }))),
            Token::Bang(operator) => Ok(Expression::Unary(Unary::Not(UnaryExpression {
                operator: *operator,
                expr: Box::new(self.advance().and_then(|_| self.primary())?),
            }))),
            _ => self.primary(),
        }
    }

    fn primary(&mut self) -> ParseResult<Expression> {
        self.advance().and_then(|token| match token {
            Token::LeftParen(start) => {
                let expr = Box::new(self.expression()?);
                let end = self.take_right_paren()?;
                Ok(Expression::Primary(Primary::Group(GroupedExpression {
                    start,
                    end,
                    expr,
                })))
            }
            Token::Number(t) => Ok(Expression::Primary(Primary::Number(t))),
            Token::String(t) => Ok(Expression::Primary(Primary::String(t))),
            Token::Identifier(t) => Ok(Expression::Primary(Primary::Ident(t))),
            Token::Nil(t) => Ok(Expression::Primary(Primary::Nil(t))),
            Token::True(t) => Ok(Expression::Primary(Primary::True(t))),
            Token::False(t) => Ok(Expression::Primary(Primary::False(t))),
            _ => Err(ParseError::UnexpectedToken(token)),
        })
    }

    fn take_right_paren(&mut self) -> ParseResult<Span> {
        self.advance().and_then(|token| match token {
            Token::RightParen(t) => Ok(t),
            _ => Err(ParseError::WrongToken(WrongToken {
                wanted: ")",
                actual: token,
            })),
        })
    }

    fn peek_next(&mut self) -> ParseResult<&Token> {
        match self.scanner.peek() {
            Some(Ok(token)) => Ok(token),
            Some(Err(e)) => Err(ParseError::Scan(*e)),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn advance(&mut self) -> ParseResult<Token> {
        match self.scanner.next() {
            Some(Ok(token)) => Ok(token),
            Some(Err(e)) => Err(ParseError::Scan(e)),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn finished(&mut self) -> bool {
        self.scanner.peek().is_none()
    }
}

/// All error states that can arise when parsing tokens.
#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    Scan(ScanError),
    WrongToken(WrongToken),
    UnexpectedToken(Token),
    UnexpectedEof,
}

/// Describes the actual [token](crate::scanner::Token) received and a `&str` representing the
/// token that was expected.
#[derive(Debug, PartialEq, Eq)]
pub struct WrongToken {
    wanted: &'static str,
    actual: Token,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scan(e) => e.fmt(f),
            Self::UnexpectedToken(t) => write!(f, "unexpected token {t}"),
            Self::WrongToken(WrongToken { wanted, actual }) => {
                write!(f, "unexpected token {actual}, wanted '{wanted}'")
            }
            Self::UnexpectedEof => f.write_str("unexpected eof"),
        }
    }
}

impl Error for ParseError {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::scanner::{Scanner, Span};

    #[test]
    fn parse_number_literal() {
        let s = "4.8";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Primary(Primary::Number(
                Span {
                    start: 0,
                    length: 3,
                    line: 1
                }
            )))))
        );
    }

    #[test]
    fn parse_string_literal() {
        let s = "\"hello\"";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Primary(Primary::String(
                Span {
                    start: 0,
                    length: 7,
                    line: 1
                }
            )))))
        );
    }

    #[test]
    fn parse_identifier() {
        let s = "x";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Primary(Primary::Ident(
                Span {
                    start: 0,
                    length: 1,
                    line: 1
                }
            )))))
        );
    }

    #[test]
    fn parse_nil() {
        let s = "nil";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Primary(Primary::Nil(
                Span {
                    start: 0,
                    length: 3,
                    line: 1
                }
            )))))
        );
    }

    #[test]
    fn parse_true() {
        let s = "true";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Primary(Primary::True(
                Span {
                    start: 0,
                    length: 4,
                    line: 1
                }
            )))))
        );
    }

    #[test]
    fn parse_false() {
        let s = "false";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Primary(Primary::False(
                Span {
                    start: 0,
                    length: 5,
                    line: 1
                }
            )))))
        );
    }

    #[test]
    fn parse_unary_negate() {
        let s = "-9";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Unary(Unary::Negate(
                UnaryExpression {
                    operator: Span {
                        start: 0,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 1,
                        length: 1,
                        line: 1
                    })))
                }
            )))))
        );
    }

    #[test]
    fn parse_unary_not() {
        let s = "!y";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Unary(Unary::Not(
                UnaryExpression {
                    operator: Span {
                        start: 0,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Primary(Primary::Ident(Span {
                        start: 1,
                        length: 1,
                        line: 1
                    })))
                }
            )))))
        );
    }

    #[test]
    fn parse_division() {
        let s = "8 / 4 / 2";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Factor(Factor::Divide(
                BinaryExpression {
                    operator: Span {
                        start: 6,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Factor(Factor::Divide(BinaryExpression {
                        operator: Span {
                            start: 2,
                            length: 1,
                            line: 1,
                        },
                        left: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 0,
                            length: 1,
                            line: 1,
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 4,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 8,
                        length: 1,
                        line: 1
                    }))),
                }
            )))))
        );
    }

    #[test]
    fn parse_multiplication() {
        let s = "8 * 4 * 2";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Factor(Factor::Multiply(
                BinaryExpression {
                    operator: Span {
                        start: 6,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Factor(Factor::Multiply(BinaryExpression {
                        operator: Span {
                            start: 2,
                            length: 1,
                            line: 1,
                        },
                        left: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 0,
                            length: 1,
                            line: 1,
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 4,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 8,
                        length: 1,
                        line: 1
                    }))),
                }
            )))))
        );
    }

    #[test]
    fn parse_unary_factor_precedence() {
        let s = "-8 * -9";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Factor(Factor::Multiply(
                BinaryExpression {
                    operator: Span {
                        start: 3,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Unary(Unary::Negate(UnaryExpression {
                        operator: Span {
                            start: 0,
                            length: 1,
                            line: 1,
                        },
                        expr: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 1,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                    right: Box::new(Expression::Unary(Unary::Negate(UnaryExpression {
                        operator: Span {
                            start: 5,
                            length: 1,
                            line: 1,
                        },
                        expr: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 6,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                }
            )))))
        );
    }

    #[test]
    fn parse_subtraction() {
        let s = "8 - 4 - 2";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Term(Term::Minus(
                BinaryExpression {
                    operator: Span {
                        start: 6,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Term(Term::Minus(BinaryExpression {
                        operator: Span {
                            start: 2,
                            length: 1,
                            line: 1,
                        },
                        left: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 0,
                            length: 1,
                            line: 1,
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 4,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 8,
                        length: 1,
                        line: 1
                    }))),
                }
            )))))
        );
    }

    #[test]
    fn parse_addition() {
        let s = "8 + 4 + 2";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Term(Term::Plus(
                BinaryExpression {
                    operator: Span {
                        start: 6,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Term(Term::Plus(BinaryExpression {
                        operator: Span {
                            start: 2,
                            length: 1,
                            line: 1,
                        },
                        left: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 0,
                            length: 1,
                            line: 1,
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 4,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 8,
                        length: 1,
                        line: 1
                    }))),
                }
            )))))
        );
    }

    #[test]
    fn parse_term_factor_precedence() {
        let s = "8 + 4 / 2";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Term(Term::Plus(
                BinaryExpression {
                    operator: Span {
                        start: 2,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Factor(Factor::Divide(BinaryExpression {
                        operator: Span {
                            start: 6,
                            length: 1,
                            line: 1,
                        },
                        left: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 4,
                            length: 1,
                            line: 1,
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 8,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                }
            )))))
        );
    }

    #[test]
    fn parse_comparison_less() {
        let s = "8 < 4";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Comparison(
                Comparison::LessThan(BinaryExpression {
                    operator: Span {
                        start: 2,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 4,
                        length: 1,
                        line: 1,
                    }))),
                })
            ))))
        );
    }

    #[test]
    fn parse_comparison_less_or_equals() {
        let s = "8 <= 4";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Comparison(
                Comparison::LessThanOrEquals(BinaryExpression {
                    operator: Span {
                        start: 2,
                        length: 2,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 5,
                        length: 1,
                        line: 1,
                    }))),
                })
            ))))
        );
    }

    #[test]
    fn parse_comparison_greater() {
        let s = "8 > 4";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Comparison(
                Comparison::GreaterThan(BinaryExpression {
                    operator: Span {
                        start: 2,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 4,
                        length: 1,
                        line: 1,
                    }))),
                })
            ))))
        );
    }

    #[test]
    fn parse_comparison_greater_or_equals() {
        let s = "8 >= 4";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Comparison(
                Comparison::GreaterThanOrEquals(BinaryExpression {
                    operator: Span {
                        start: 2,
                        length: 2,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 5,
                        length: 1,
                        line: 1,
                    }))),
                })
            ))))
        );
    }

    #[test]
    fn parse_comparison_term_precedence() {
        let s = "8 > 4 - 2";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Comparison(
                Comparison::GreaterThan(BinaryExpression {
                    operator: Span {
                        start: 2,
                        length: 1,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Term(Term::Minus(BinaryExpression {
                        operator: Span {
                            start: 6,
                            length: 1,
                            line: 1,
                        },
                        left: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 4,
                            length: 1,
                            line: 1,
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 8,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                })
            ))))
        );
    }

    #[test]
    fn parse_equality() {
        let s = "8 == 4";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Equality(Equality::Equals(
                BinaryExpression {
                    operator: Span {
                        start: 2,
                        length: 2,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 5,
                        length: 1,
                        line: 1,
                    }))),
                }
            )))))
        );
    }

    #[test]
    fn parse_inequality() {
        let s = "8 != 4";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Equality(
                Equality::NotEquals(BinaryExpression {
                    operator: Span {
                        start: 2,
                        length: 2,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 5,
                        length: 1,
                        line: 1,
                    }))),
                })
            ))))
        );
    }

    #[test]
    fn parse_equality_comparison_precedence() {
        let s = "true != 4 > 2";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Equality(
                Equality::NotEquals(BinaryExpression {
                    operator: Span {
                        start: 5,
                        length: 2,
                        line: 1
                    },
                    left: Box::new(Expression::Primary(Primary::True(Span {
                        start: 0,
                        length: 4,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Comparison(Comparison::GreaterThan(
                        BinaryExpression {
                            operator: Span {
                                start: 10,
                                length: 1,
                                line: 1,
                            },
                            left: Box::new(Expression::Primary(Primary::Number(Span {
                                start: 8,
                                length: 1,
                                line: 1,
                            }))),
                            right: Box::new(Expression::Primary(Primary::Number(Span {
                                start: 12,
                                length: 1,
                                line: 1,
                            }))),
                        }
                    ))),
                })
            ))))
        );
    }

    #[test]
    fn parse_grouped() {
        let s = "(1 + 2)";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Statement::Expr(Expression::Primary(Primary::Group(
                GroupedExpression {
                    start: Span {
                        start: 0,
                        length: 1,
                        line: 1
                    },
                    end: Span {
                        start: 6,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Term(Term::Plus(BinaryExpression {
                        operator: Span {
                            start: 3,
                            length: 1,
                            line: 1,
                        },
                        left: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 1,
                            length: 1,
                            line: 1,
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 5,
                            length: 1,
                            line: 1,
                        }))),
                    }))),
                }
            )))))
        );
    }
}
