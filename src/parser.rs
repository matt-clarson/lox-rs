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
            _ => self.statement().map(Declaration::Stmt),
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

        let mut body = self.block()?;
        let has_return = matches!(body.last(), Some(Declaration::Stmt(Statement::Return(_))));
        if !has_return {
            body.push(Declaration::Stmt(Statement::Return(None)));
        }
        Ok(Declaration::Function {
            name,
            args: args.into(),
            body: body.into(),
        })
    }

    fn var_declaration(&mut self) -> ParseResult<Declaration> {
        let var = self.var()?;
        self.take_semicolon().and(Ok(Declaration::Var(var)))
    }

    fn statement(&mut self) -> ParseResult<Statement> {
        match self.current()? {
            Token::Print(_) => {
                self.advance()?;
                let stmt = Statement::Print(self.expression()?);
                self.take_semicolon().and(Ok(stmt))
            }
            Token::Return(_) => {
                self.advance()?;
                let next_is_semicolon = matches!(self.current()?, Token::Semicolon(_));
                if next_is_semicolon {
                    return self.advance().and(Ok(Statement::Return(None)));
                }
                let stmt = Statement::Return(Some(self.expression()?));
                self.take_semicolon().and(Ok(stmt))
            }
            Token::If(_) => {
                self.advance()?;
                self.take_left_paren()?;
                let condition = self.expression()?;
                self.take_right_paren()?;
                let body = self.statement()?.into();

                let has_else = !self.finished() && matches!(self.current()?, Token::Else(_));
                if !has_else {
                    return Ok(Statement::If {
                        condition,
                        body,
                        else_body: None,
                    });
                }

                self.advance()?;

                let else_body = self.statement()?.into();

                Ok(Statement::If {
                    condition,
                    body,
                    else_body: Some(else_body),
                })
            }
            Token::While(_) => {
                self.advance()?;
                self.take_left_paren()?;
                let condition = self.expression()?;
                self.take_right_paren()?;

                let body = self.statement()?.into();

                Ok(Statement::While { condition, body })
            }
            Token::For(_) => {
                self.advance()?;
                self.take_left_paren()?;
                let initialiser = match self.current()? {
                    Token::Var(_) => {
                        self.advance()?;
                        let var = self.var()?;
                        self.take_semicolon()?;
                        Some(ForInitialiser::Var(var))
                    }
                    Token::Semicolon(_) => {
                        self.advance()?;
                        None
                    }
                    _ => {
                        let expr = self.expression()?;
                        self.take_semicolon()?;
                        Some(ForInitialiser::Expr(expr))
                    }
                };

                let condition = if let Token::Semicolon(_) = self.current()? {
                    self.advance()?;
                    None
                } else {
                    let expr = self.expression()?;
                    self.take_semicolon()?;
                    Some(expr)
                };

                let incrementer = if let Token::RightParen(_) = self.current()? {
                    self.advance()?;
                    None
                } else {
                    let expr = self.expression()?;
                    self.take_right_paren()?;
                    Some(expr)
                };

                let body = self.statement()?.into();

                Ok(Statement::For {
                    initialiser,
                    condition,
                    incrementer,
                    body,
                })
            }
            Token::LeftBrace(_) => self.block().map(Into::into).map(Statement::Block),
            _ => {
                let stmt = Statement::Expr(self.expression()?);
                self.take_semicolon().and(Ok(stmt))
            }
        }
    }

    fn block(&mut self) -> ParseResult<Vec<Declaration>> {
        self.take_left_brace()?;
        let mut contents = vec![];

        loop {
            let next_is_right_brace = matches!(self.current()?, Token::RightBrace(_));
            if next_is_right_brace {
                self.advance()?;
                return Ok(contents);
            }

            contents.push(self.declaration()?);
        }
    }

    fn expression(&mut self) -> ParseResult<Expression> {
        match self.curr_and_next()? {
            (Token::Identifier(ident), Some(Token::Equal(_))) => {
                self.advance().and_then(|_| self.advance())?;
                let expr = Box::new(self.expression()?);
                Ok(Expression::Assignment { ident, expr })
            }
            _ => self.or(),
        }
    }

    fn or(&mut self) -> ParseResult<Expression> {
        let mut expr = self.and()?;

        while let Token::Or(_) = self.current()? {
            self.advance()?;
            let right = self.and()?.into();
            expr = Expression::Or {
                left: expr.into(),
                right,
            }
        }

        Ok(expr)
    }

    fn and(&mut self) -> ParseResult<Expression> {
        let mut expr = self.equality()?;

        while let Token::And(_) = self.current()? {
            self.advance()?;
            let right = self.equality()?.into();
            expr = Expression::And {
                left: expr.into(),
                right,
            }
        }

        Ok(expr)
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
            expr = Expression::Call {
                callee: expr.into(),
                args,
            };
        }

        Ok(expr)
    }

    fn call_args(&mut self) -> Result<Option<Box<[Expression]>>, ParseError> {
        let mut args = vec![];

        let next_is_left_paren = matches!(self.current()?, Token::LeftParen(_));
        if !next_is_left_paren {
            return Ok(None);
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

    fn var(&mut self) -> ParseResult<Var> {
        let name = self.take_ident()?;
        let expr = if let Token::Equal(_) = self.current()? {
            self.advance()?;
            self.expression()?
        } else {
            Expression::Primary(Primary::Nil)
        };

        Ok(Var { name, expr })
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
            Some(Ok(Declaration::Var(Var {
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
            })))
        );
    }

    #[test]
    fn parse_var_declaration() {
        let s = "var x;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Var(Var {
                name: Span {
                    start: 4,
                    length: 1,
                    line: 1
                },
                expr: Expression::Primary(Primary::Nil)
            })))
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
                    Declaration::Var(Var {
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
                    }),
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

    #[test]
    fn parse_if_statement() {
        let s = "
if (true) {
    print \"hello\";
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::If {
                condition: Expression::Primary(Primary::True),
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::String(Span {
                            start: 22,
                            length: 7,
                            line: 2
                        })
                    )))]
                    .into()
                )),
                else_body: None
            })))
        );
    }

    #[test]
    fn parse_if_statement_no_block() {
        let s = "if (true) return 1;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::If {
                condition: Expression::Primary(Primary::True),
                body: Box::new(Statement::Return(Some(Expression::Primary(
                    Primary::Number(Span {
                        start: 17,
                        length: 1,
                        line: 1
                    })
                )))),
                else_body: None
            })))
        );
    }

    #[test]
    fn parse_if_else_statement() {
        let s = "
if (true) {
    print \"hello\";
} else {
    print \"goodbye\";
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::If {
                condition: Expression::Primary(Primary::True),
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::String(Span {
                            start: 22,
                            length: 7,
                            line: 2
                        })
                    )))]
                    .into()
                )),
                else_body: Some(Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::String(Span {
                            start: 50,
                            length: 9,
                            line: 4
                        })
                    )))]
                    .into()
                )))
            })))
        );
    }

    #[test]
    fn parse_or_expression() {
        let s = "true or false;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::Or {
                left: Box::new(Expression::Primary(Primary::True)),
                right: Box::new(Expression::Primary(Primary::False))
            }))))
        );
    }

    #[test]
    fn parse_and_expression() {
        let s = "true and false;";

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::Expr(Expression::And {
                left: Box::new(Expression::Primary(Primary::True)),
                right: Box::new(Expression::Primary(Primary::False))
            }))))
        );
    }

    #[test]
    fn parse_while_loop() {
        let s = "
while (true) {
    print \"loop\";
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::While {
                condition: Expression::Primary(Primary::True),
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::String(Span {
                            start: 25,
                            length: 6,
                            line: 2
                        })
                    )))]
                    .into()
                ))
            })))
        );
    }

    #[test]
    fn parse_common_for_loop() {
        let s = "
for (var i=0; i<5; i=i+1) {
    print i;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::For {
                initialiser: Some(ForInitialiser::Var(Var {
                    name: Span {
                        start: 9,
                        length: 1,
                        line: 1
                    },
                    expr: Expression::Primary(Primary::Number(Span {
                        start: 11,
                        length: 1,
                        line: 1
                    }))
                })),
                condition: Some(Expression::Comparison(Comparison::LessThan {
                    left: Box::new(Expression::Primary(Primary::Ident(Span {
                        start: 14,
                        length: 1,
                        line: 1
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 16,
                        length: 1,
                        line: 1
                    })))
                })),
                incrementer: Some(Expression::Assignment {
                    ident: Span {
                        start: 19,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Term(Term::Plus {
                        left: Box::new(Expression::Primary(Primary::Ident(Span {
                            start: 21,
                            length: 1,
                            line: 1
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 23,
                            length: 1,
                            line: 1
                        })))
                    }))
                }),
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::Ident(Span {
                            start: 38,
                            length: 1,
                            line: 2
                        })
                    )))]
                    .into()
                ))
            })))
        );
    }

    #[test]
    fn parse_for_loop_expression_initialiser() {
        let s = "
for (i=0; i<5; i=i+1) {
    print i;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::For {
                initialiser: Some(ForInitialiser::Expr(Expression::Assignment{
                    ident: Span{
                        start: 5, length: 1, line: 1
                    },
                    expr: Box::new(Expression::Primary(Primary::Number(Span{
                        start: 7, length: 1, line: 1
                    })))
                })),
                condition: Some(Expression::Comparison(Comparison::LessThan {
                    left: Box::new(Expression::Primary(Primary::Ident(Span {
                        start: 10,
                        length: 1,
                        line: 1
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 12,
                        length: 1,
                        line: 1
                    })))
                })),
                incrementer: Some(Expression::Assignment {
                    ident: Span {
                        start: 15,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Term(Term::Plus {
                        left: Box::new(Expression::Primary(Primary::Ident(Span {
                            start: 17,
                            length: 1,
                            line: 1
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 19,
                            length: 1,
                            line: 1
                        })))
                    }))
                }),
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::Ident(Span {
                            start: 34,
                            length: 1,
                            line: 2
                        })
                    )))]
                    .into()
                ))
            })))
        );
    }

    #[test]
    fn parse_for_loop_no_initialiser() {
        let s = "
for (; i<5; i=i+1) {
    print i;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::For {
                initialiser: None,
                condition: Some(Expression::Comparison(Comparison::LessThan {
                    left: Box::new(Expression::Primary(Primary::Ident(Span {
                        start: 7,
                        length: 1,
                        line: 1
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 9,
                        length: 1,
                        line: 1
                    })))
                })),
                incrementer: Some(Expression::Assignment {
                    ident: Span {
                        start: 12,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Term(Term::Plus {
                        left: Box::new(Expression::Primary(Primary::Ident(Span {
                            start: 14,
                            length: 1,
                            line: 1
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 16,
                            length: 1,
                            line: 1
                        })))
                    }))
                }),
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::Ident(Span {
                            start: 31,
                            length: 1,
                            line: 2
                        })
                    )))]
                    .into()
                ))
            })))
        );
    }

    #[test]
    fn parse_for_loop_no_condition() {
        let s = "
for (var i=0;; i=i+1) {
    print i;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::For {
                initialiser: Some(ForInitialiser::Var(Var {
                    name: Span {
                        start: 9,
                        length: 1,
                        line: 1
                    },
                    expr: Expression::Primary(Primary::Number(Span {
                        start: 11,
                        length: 1,
                        line: 1
                    }))
                })),
                condition: None,
                incrementer: Some(Expression::Assignment {
                    ident: Span {
                        start: 15,
                        length: 1,
                        line: 1
                    },
                    expr: Box::new(Expression::Term(Term::Plus {
                        left: Box::new(Expression::Primary(Primary::Ident(Span {
                            start: 17,
                            length: 1,
                            line: 1
                        }))),
                        right: Box::new(Expression::Primary(Primary::Number(Span {
                            start: 19,
                            length: 1,
                            line: 1
                        })))
                    }))
                }),
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::Ident(Span {
                            start: 34,
                            length: 1,
                            line: 2
                        })
                    )))]
                    .into()
                ))
            })))
        );
    }

    #[test]
    fn parse_for_loop_no_incrementer() {
        let s = "
for (var i=0; i<5;) {
    print i;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::For {
                initialiser: Some(ForInitialiser::Var(Var {
                    name: Span {
                        start: 9,
                        length: 1,
                        line: 1
                    },
                    expr: Expression::Primary(Primary::Number(Span {
                        start: 11,
                        length: 1,
                        line: 1
                    }))
                })),
                condition: Some(Expression::Comparison(Comparison::LessThan {
                    left: Box::new(Expression::Primary(Primary::Ident(Span {
                        start: 14,
                        length: 1,
                        line: 1
                    }))),
                    right: Box::new(Expression::Primary(Primary::Number(Span {
                        start: 16,
                        length: 1,
                        line: 1
                    })))
                })),
                incrementer: None,
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::Ident(Span {
                            start: 32,
                            length: 1,
                            line: 2
                        })
                    )))]
                    .into()
                ))
            })))
        );
    }

    #[test]
    fn parse_empty_for_loop() {
        let s = "
for (;;) {
    print i;
}
        "
        .trim();

        let scanner = Scanner::from(s);

        let mut parser = Parser::from(scanner);

        assert_eq!(
            parser.next(),
            Some(Ok(Declaration::Stmt(Statement::For {
                initialiser: None,
                condition: None,
                incrementer: None,
                body: Box::new(Statement::Block(
                    vec![Declaration::Stmt(Statement::Print(Expression::Primary(
                        Primary::Ident(Span {
                            start: 21,
                            length: 1,
                            line: 2
                        })
                    )))]
                    .into()
                ))
            })))
        );
    }
}
