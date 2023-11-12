use std::{error::Error, fmt::Display, iter::Peekable};

use crate::{
    ast::*,
    scanner::{ScanError, Scanner, Span, Token},
};

/// Uses a source of scanned [tokens](crate::scanner::Token) to output an AST, as a stream of
/// [declarations](crate::ast::Declaration).
pub struct Parser<'s> {
    scanner: Peekable<Scanner<'s>>,
    curr: Option<Result<Token, ScanError>>,
}

impl<'s> From<Scanner<'s>> for Parser<'s> {
    fn from(scanner: Scanner<'s>) -> Self {
        let mut scanner = scanner.peekable();
        let curr = scanner.next();
        Self { scanner, curr }
    }
}

impl<'s> Iterator for Parser<'s> {
    type Item = Result<Declaration, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished() {
            None
        } else {
            Some(self.declaration())
        }
    }
}

type ParseResult<T> = Result<T, ParseError>;

impl<'s> Parser<'s> {
    fn declaration(&mut self) -> ParseResult<Declaration> {
        match self.current()? {
            Token::Fun(_) => {
                self.advance()?;
                self.function_declaration()
            }
            Token::Var(_) => {
                self.advance()?;
                self.var_declaration()
            }
            _ => self.statement(),
        }
    }

    fn function_declaration(&mut self) -> ParseResult<Declaration> {
        let name = self.take_ident()?;
        self.take_left_paren()?;
        let mut args = vec![];
        loop {
            if let Token::Identifier(span) = self.current()? {
                self.advance()?;
                args.push(span);
            }
            if let Token::Comma(_) = self.current()? {
                self.advance()?;
                continue;
            } else {
                break;
            }
        }
        self.take_right_paren()?;
        self.take_left_brace()?;

        let mut body = vec![];

        loop {
            match self.current()? {
                Token::RightBrace(_) => {
                    let has_return =
                        matches!(body.last(), Some(Declaration::Stmt(Statement::Return(_))));
                    if !has_return {
                        body.push(Declaration::Stmt(Statement::Return(None)));
                    }
                    let decl = Declaration::Function {
                        name,
                        args: args.into(),
                        body: body.into(),
                    };
                    return self.advance().and(Ok(decl));
                }
                Token::Return(_) => {
                    self.advance()?;
                    let decl = match self.current()? {
                        Token::Semicolon(_) => {
                            let decl = Declaration::Stmt(Statement::Return(None));
                            self.advance()?;
                            decl
                        }
                        _ => {
                            let expr = self.expression()?;
                            self.take_semicolon()?;
                            Declaration::Stmt(Statement::Return(Some(expr)))
                        }
                    };
                    body.push(decl);
                }
                _ => body.push(self.declaration()?),
            }
        }
    }

    fn var_declaration(&mut self) -> ParseResult<Declaration> {
        let name = self.take_ident()?;
        let expr = if let Token::Equal(_) = self.current()? {
            self.advance()?;
            self.expression()?
        } else {
            Expression::Primary(Primary::Nil)
        };

        let decl = Declaration::Var { name, expr };
        self.take_semicolon().and(Ok(decl))
    }

    fn statement(&mut self) -> ParseResult<Declaration> {
        match self.current()? {
            Token::Print(_) => {
                self.advance()?;
                let stmt = Declaration::Stmt(Statement::Print(self.expression()?));
                self.take_semicolon().and(Ok(stmt))
            }
            Token::LeftBrace(_) => {
                self.advance()?;
                let mut contents: Vec<Declaration> = vec![];
                loop {
                    match self.current()? {
                        Token::RightBrace(_) => {
                            let stmt = Declaration::Stmt(Statement::Block(contents.into()));
                            return self.advance().and(Ok(stmt));
                        }
                        _ => {
                            contents.push(self.declaration()?);
                        }
                    }
                }
            }
            _ => {
                let stmt = Declaration::Stmt(Statement::Expr(self.expression()?));
                self.take_semicolon().and(Ok(stmt))
            }
        }
    }

    fn expression(&mut self) -> ParseResult<Expression> {
        match self.curr_and_next()? {
            (Token::Identifier(ident), Some(Token::Equal(_))) => {
                self.advance().and_then(|_| self.advance())?;
                let expr = Box::new(self.expression()?);
                Ok(Expression::Assignment { ident, expr })
            }
            _ => self.equality(),
        }
    }

    fn equality(&mut self) -> ParseResult<Expression> {
        let mut expr = self.comparison()?;

        loop {
            expr = match self.current() {
                Ok(Token::EqualEqual(_)) => Expression::Equality(Equality::Equals {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.comparison())?),
                }),
                Ok(Token::BangEqual(_)) => Expression::Equality(Equality::NotEquals {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.comparison())?),
                }),
                _ => break,
            };
        }

        Ok(expr)
    }

    fn comparison(&mut self) -> ParseResult<Expression> {
        let mut expr = self.term()?;

        loop {
            expr = match self.current() {
                Ok(Token::Less(_)) => Expression::Comparison(Comparison::LessThan {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.term())?),
                }),
                Ok(Token::LessEqual(_)) => Expression::Comparison(Comparison::LessThanOrEquals {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.term())?),
                }),
                Ok(Token::Greater(_)) => Expression::Comparison(Comparison::GreaterThan {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.term())?),
                }),
                Ok(Token::GreaterEqual(_)) => {
                    Expression::Comparison(Comparison::GreaterThanOrEquals {
                        left: Box::new(expr),
                        right: Box::new(self.advance().and_then(|_| self.term())?),
                    })
                }
                _ => break,
            };
        }

        Ok(expr)
    }

    fn term(&mut self) -> ParseResult<Expression> {
        let mut expr = self.factor()?;

        loop {
            expr = match self.current() {
                Ok(Token::Minus(_)) => Expression::Term(Term::Minus {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.factor())?),
                }),
                Ok(Token::Plus(_)) => Expression::Term(Term::Plus {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.factor())?),
                }),
                _ => break,
            };
        }

        Ok(expr)
    }

    fn factor(&mut self) -> ParseResult<Expression> {
        let mut expr = self.unary()?;

        loop {
            expr = match self.current() {
                Ok(Token::Slash(_)) => Expression::Factor(Factor::Divide {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.unary())?),
                }),
                Ok(Token::Star(_)) => Expression::Factor(Factor::Multiply {
                    left: Box::new(expr),
                    right: Box::new(self.advance().and_then(|_| self.unary())?),
                }),
                _ => break,
            };
        }

        Ok(expr)
    }

    fn unary(&mut self) -> ParseResult<Expression> {
        match self.current()? {
            Token::Minus(_) => Ok(Expression::Unary(Unary::Negate(Box::new(
                self.advance().and_then(|_| self.call())?,
            )))),
            Token::Bang(_) => Ok(Expression::Unary(Unary::Not(Box::new(
                self.advance().and_then(|_| self.call())?,
            )))),
            _ => self.call(),
        }
    }

    fn call(&mut self) -> ParseResult<Expression> {
        let mut expr = self.primary()?;

        while let Some(args) = self.call_args()? {
            expr = Expression::Call{
                callee: expr.into(),
                args
            };
        }

        Ok(expr)
    }

    fn call_args(&mut self) -> Result<Option<Box<[Expression]>>, ParseError> {
        let mut args = vec![];

        let next_is_left_paren = matches!(self.current()?, Token::LeftParen(_));
        if !next_is_left_paren {
            return Ok(None)
        }

        self.advance()?;

        let next_is_right_paren = matches!(self.current()?, Token::RightParen(_));
        if next_is_right_paren {
            self.advance()?;
            return Ok(Some(args.into()));
        }

        loop {
            args.push(self.expression()?);

            let next_is_comma = matches!(self.current()?, Token::Comma(_));
            if !next_is_comma {
                break;
            }
            self.advance()?
        }

        self.take_right_paren()?;

        Ok(Some(args.into()))
    }

    fn primary(&mut self) -> ParseResult<Expression> {
        match self.current()? {
            Token::LeftParen(_) => {
                self.advance()?;
                let expr = Expression::Primary(Primary::Group(Box::new(self.expression()?)));
                self.take_right_paren().and(Ok(expr))
            }
            Token::Number(t) => self
                .advance()
                .and(Ok(Expression::Primary(Primary::Number(t)))),
            Token::String(t) => self
                .advance()
                .and(Ok(Expression::Primary(Primary::String(t)))),
            Token::Identifier(t) => self
                .advance()
                .and(Ok(Expression::Primary(Primary::Ident(t)))),
            Token::Nil(_) => self.advance().and(Ok(Expression::Primary(Primary::Nil))),
            Token::True(_) => self.advance().and(Ok(Expression::Primary(Primary::True))),
            Token::False(_) => self.advance().and(Ok(Expression::Primary(Primary::False))),
            token => Err(ParseError::UnexpectedToken(token)),
        }
    }

    fn take_ident(&mut self) -> ParseResult<Span> {
        match self.current()? {
            Token::Identifier(t) => self.advance().and(Ok(t)),
            token => Err(ParseError::WrongToken(WrongToken {
                wanted: "[identifier]",
                actual: token,
            })),
        }
    }

    fn take_semicolon(&mut self) -> ParseResult<Span> {
        match self.current()? {
            Token::Semicolon(t) => self.advance().and(Ok(t)),
            token => Err(ParseError::WrongToken(WrongToken {
                wanted: ";",
                actual: token,
            })),
        }
    }

    fn take_left_paren(&mut self) -> ParseResult<Span> {
        match self.current()? {
            Token::LeftParen(t) => self.advance().and(Ok(t)),
            token => Err(ParseError::WrongToken(WrongToken {
                wanted: "(",
                actual: token,
            })),
        }
    }

    fn take_right_paren(&mut self) -> ParseResult<Span> {
        match self.current()? {
            Token::RightParen(t) => self.advance().and(Ok(t)),
            token => Err(ParseError::WrongToken(WrongToken {
                wanted: ")",
                actual: token,
            })),
        }
    }

    fn take_left_brace(&mut self) -> ParseResult<Span> {
        match self.current()? {
            Token::LeftBrace(t) => self.advance().and(Ok(t)),
            token => Err(ParseError::WrongToken(WrongToken {
                wanted: "{",
                actual: token,
            })),
        }
    }

    fn advance(&mut self) -> ParseResult<()> {
        self.curr = self.scanner.next();
        Ok(())
    }

    fn curr_and_next(&mut self) -> ParseResult<(Token, Option<&Token>)> {
        self.current().and_then(|curr| match self.scanner.peek() {
            Some(Ok(next)) => Ok((curr, Some(next))),
            None => Ok((curr, None)),
            Some(Err(e)) => Err(e.into()),
        })
    }

    fn current(&self) -> ParseResult<Token> {
        match self.curr {
            Some(Ok(token)) => Ok(token),
            Some(Err(e)) => Err(ParseError::Scan(e)),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn finished(&mut self) -> bool {
        self.scanner.peek().or(self.curr.as_ref()).is_none()
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

impl From<ScanError> for ParseError {
    fn from(value: ScanError) -> Self {
        Self::Scan(value)
    }
}

impl From<&ScanError> for ParseError {
    fn from(value: &ScanError) -> Self {
        Self::Scan(*value)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::scanner::{Scanner, Span};

    #[test]
    fn parse_number_literal() {
        let s = "4.8;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Primary(
                Primary::Number(Span {
                    start: 0,
                    length: 3,
                    line: 1
                })
            )))))
        );
    }

    #[test]
    fn parse_string_literal() {
        let s = "\"hello\";";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Primary(
                Primary::String(Span {
                    start: 0,
                    length: 7,
                    line: 1
                })
            )))))
        );
    }

    #[test]
    fn parse_identifier() {
        let s = "x;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Primary(
                Primary::Ident(Span {
                    start: 0,
                    length: 1,
                    line: 1
                })
            )))))
        );
    }

    #[test]
    fn parse_nil() {
        let s = "nil;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Primary(
                Primary::Nil
            )))))
        );
    }

    #[test]
    fn parse_true() {
        let s = "true;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Primary(
                Primary::True
            )))))
        );
    }

    #[test]
    fn parse_false() {
        let s = "false;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Primary(
                Primary::False
            )))))
        );
    }

    #[test]
    fn parse_unary_negate() {
        let s = "-9;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Unary(
                Unary::Negate(Box::new(Expression::Primary(Primary::Number(Span {
                    start: 1,
                    length: 1,
                    line: 1
                }))))
            )))))
        );
    }

    #[test]
    fn parse_unary_not() {
        let s = "!y;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Unary(
                Unary::Not(Box::new(Expression::Primary(Primary::Ident(Span {
                    start: 1,
                    length: 1,
                    line: 1
                }))))
            )))))
        );
    }

    #[test]
    fn parse_division() {
        let s = "8 / 4 / 2;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Factor(
                Factor::Divide {
                    left: Box::new(Expression::Factor(Factor::Divide {
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
                    })),
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
        let s = "8 * 4 * 2;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Factor(
                Factor::Multiply {
                    left: Box::new(Expression::Factor(Factor::Multiply {
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
                    })),
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
        let s = "-8 * -9;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Factor(
                Factor::Multiply {
                    left: Box::new(Expression::Unary(Unary::Negate(Box::new(
                        Expression::Primary(Primary::Number(Span {
                            start: 1,
                            length: 1,
                            line: 1,
                        }))
                    ),))),
                    right: Box::new(Expression::Unary(Unary::Negate(Box::new(
                        Expression::Primary(Primary::Number(Span {
                            start: 6,
                            length: 1,
                            line: 1,
                        }))
                    ),))),
                }
            )))))
        );
    }

    #[test]
    fn parse_subtraction() {
        let s = "8 - 4 - 2;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Term(
                Term::Minus {
                    left: Box::new(Expression::Term(Term::Minus {
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
                    })),
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
        let s = "8 + 4 + 2;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Term(
                Term::Plus {
                    left: Box::new(Expression::Term(Term::Plus {
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
                    })),
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
        let s = "8 + 4 / 2;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Term(
                Term::Plus {
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Factor(Factor::Divide {
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
                    })),
                }
            )))))
        );
    }

    #[test]
    fn parse_comparison_less() {
        let s = "8 < 4;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Comparison(Comparison::LessThan {
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
        let s = "8 <= 4;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Comparison(Comparison::LessThanOrEquals {
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
        let s = "8 > 4;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Comparison(Comparison::GreaterThan {
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
        let s = "8 >= 4;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Comparison(Comparison::GreaterThanOrEquals {
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
        let s = "8 > 4 - 2;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Comparison(Comparison::GreaterThan {
                    left: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 0,
                        length: 1,
                        line: 1,
                    }))),
                    right: Box::new(Expression::Term(Term::Minus {
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
                    })),
                })
            ))))
        );
    }

    #[test]
    fn parse_equality() {
        let s = "8 == 4;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Equality(Equality::Equals {
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
    fn parse_inequality() {
        let s = "8 != 4;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Equality(Equality::NotEquals {
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
        let s = "true != 4 > 2;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Equality(Equality::NotEquals {
                    left: Box::new(Expression::Primary(Primary::True)),
                    right: Box::new(Expression::Comparison(Comparison::GreaterThan {
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
                    })),
                })
            ))))
        );
    }

    #[test]
    fn parse_grouped() {
        let s = "(1 + 2);";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Primary(
                Primary::Group(Box::new(Expression::Term(Term::Plus {
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
                })),)
            )))))
        );
    }

    #[test]
    fn parse_print() {
        let s = "print x;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Print(
                Expression::Primary(Primary::Ident(Span {
                    start: 6,
                    length: 1,
                    line: 1
                }))
            ))))
        );
    }

    #[test]
    fn parse_var_assignment() {
        let s = "var x = 7;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Var {
                name: Span {
                    start: 4,
                    length: 1,
                    line: 1
                },
                expr: Expression::Primary(Primary::Number(Span {
                    start: 8,
                    length: 1,
                    line: 1
                }))
            }))
        );
    }

    #[test]
    fn parse_var_declaration() {
        let s = "var x;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Var {
                name: Span {
                    start: 4,
                    length: 1,
                    line: 1
                },
                expr: Expression::Primary(Primary::Nil)
            }))
        );
    }

    #[test]
    fn parse_assignment() {
        let s = "x = 5.3;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Assignment {
                    ident: Span {
                        start: 0,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 4,
                        length: 3,
                        line: 1
                    })))
                }
            ))))
        );
    }

    #[test]
    fn parse_invalid_reassignment() {
        let s = "x * y = 5.3;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Err(ParseError::WrongToken(WrongToken {
                wanted: ";",
                actual: Token::Equal(Span {
                    start: 6,
                    length: 1,
                    line: 1
                })
            })))
        );
    }

    #[test]
    fn parse_multi_assignment() {
        let s = "x = y = 1;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(
                Expression::Assignment {
                    ident: Span {
                        start: 0,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Assignment {
                        ident: Span {
                            start: 4,
                            length: 1,
                            line: 1
                        },
                        expr: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 8,
                            length: 1,
                            line: 1
                        })))
                    })
                }
            ))))
        );
    }

    #[test]
    fn parse_block() {
        let s = "
{
    var x = 1;
    print x * 2;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Block(
                vec![
                    Declaration::Var {
                        name: Span {
                            start: 10,
                            length: 1,
                            line: 2,
                        },
                        expr: Expression::Primary(Primary::Number(Span {
                            start: 14,
                            length: 1,
                            line: 2,
                        }))
                    },
                    Declaration::Stmt(Statement::Print(Expression::Factor(Factor::Multiply {
                        left: Box::new(Expression::Primary(Primary::Ident(Span {
                            start: 27,
                            length: 1,
                            line: 3
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 31,
                            length: 1,
                            line: 3
                        })))
                    })))
                ]
                .into()
            ))))
        );
    }

    #[test]
    fn parse_empty_block() {
        let s = "{}";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Block(vec![].into()))))
        );
    }

    #[test]
    fn parse_void_function_no_args_no_locals() {
        let s = "
fun f() {
    print 123;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Function {
                name: Span {
                    start: 4,
                    length: 1,
                    line: 1,
                },
                args: vec![].into(),
                body: vec![
                    Declaration::Stmt(Statement::Print(Expression::Primary(Primary::Number(
                        Span {
                            start: 20,
                            length: 3,
                            line: 2,
                        }
                    )))),
                    Declaration::Stmt(Statement::Return(None))
                ]
                .into()
            }))
        );
    }

    #[test]
    fn parse_returning_function_no_args_no_locals() {
        let s = "
fun f() {
    return 123;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Function {
                name: Span {
                    start: 4,
                    length: 1,
                    line: 1,
                },
                args: vec![].into(),
                body: vec![Declaration::Stmt(Statement::Return(Some(
                    Expression::Primary(Primary::Number(Span {
                        start: 21,
                        length: 3,
                        line: 2,
                    }))
                ))),]
                .into()
            }))
        );
    }

    #[test]
    fn parse_returning_empty_function_no_args_no_locals() {
        let s = "
fun f() {
    return;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Function {
                name: Span {
                    start: 4,
                    length: 1,
                    line: 1,
                },
                args: vec![].into(),
                body: vec![Declaration::Stmt(Statement::Return(None)),].into()
            }))
        );
    }

    #[test]
    fn parse_void_function_one_arg_no_locals() {
        let s = "
fun f(a) {
    print a;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Function {
                name: Span {
                    start: 4,
                    length: 1,
                    line: 1,
                },
                args: vec![Span {
                    start: 6,
                    length: 1,
                    line: 1
                }]
                .into(),
                body: vec![
                    Declaration::Stmt(Statement::Print(Expression::Primary(Primary::Ident(
                        Span {
                            start: 21,
                            length: 1,
                            line: 2
                        }
                    )))),
                    Declaration::Stmt(Statement::Return(None))
                ]
                .into()
            }))
        );
    }

    #[test]
    fn parse_void_function_two_args_no_locals() {
        let s = "
fun f(a, b) {
    print a + b;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Function {
                name: Span {
                    start: 4,
                    length: 1,
                    line: 1,
                },
                args: vec![
                    Span {
                        start: 6,
                        length: 1,
                        line: 1
                    },
                    Span {
                        start: 9,
                        length: 1,
                        line: 1
                    }
                ]
                .into(),
                body: vec![
                    Declaration::Stmt(Statement::Print(Expression::Term(Term::Plus {
                        left: Box::new(Expression::Primary(Primary::Ident(Span {
                            start: 24,
                            length: 1,
                            line: 2,
                        }))),
                        right: Box::new(Expression::Primary(Primary::Ident(Span {
                            start: 28,
                            length: 1,
                            line: 2
                        })))
                    }))),
                    Declaration::Stmt(Statement::Return(None))
                ]
                .into()
            }))
        );
    }

    #[test]
    fn parse_function_call_no_args() {
        let s = "f();";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Call {
                callee: Box::new(Expression::Primary(Primary::Ident(Span {
                    start: 0,
                    length: 1,
                    line: 1
                }))),
                args: vec![].into()
            }))))
        );
    }

    #[test]
    fn parse_function_call_one_arg() {
        let s = "f(1);";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Call {
                callee: Box::new(Expression::Primary(Primary::Ident(Span {
                    start: 0,
                    length: 1,
                    line: 1
                }))),
                args: vec![Expression::Primary(Primary::Number(Span {
                    start: 2,
                    length: 1,
                    line: 1
                }))]
                .into()
            }))))
        );
    }

    #[test]
    fn parse_function_call_two_args() {
        let s = "f(1, 2);";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Call {
                callee: Box::new(Expression::Primary(Primary::Ident(Span {
                    start: 0,
                    length: 1,
                    line: 1
                }))),
                args: vec![
                    Expression::Primary(Primary::Number(Span {
                        start: 2,
                        length: 1,
                        line: 1
                    })),
                    Expression::Primary(Primary::Number(Span {
                        start: 5,
                        length: 1,
                        line: 1
                    }))
                ]
                .into()
            }))))
        );
    }

    #[test]
    fn parse_multi_function_call() {
        let s = "f()();";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Call {
                callee: Box::new(Expression::Call {
                    callee: Box::new(Expression::Primary(Primary::Ident(Span {
                        start: 0,
                        length: 1,
                        line: 1
                    }))),
                    args: vec![].into()
                }),
                args: vec![].into()
            }))))
        );
    }
}
