use std::{
    error::Error,
    fmt::Display,
    iter::{Enumerate, Peekable},
    str::Chars,
};

/// Scans an input string to produce a stream of tokens.
///
/// Note, the iterator yields `None` when the source EOF is reached.
///
/// # Examples
///
/// ```
/// use lox::scanner::{Scanner, Token};
///
/// let source = "var x = 50 - 8;";
///
/// let mut scanner = Scanner::from(source);
///
/// assert!(matches!(scanner.next(), Some(Ok(Token::Var(_)))));
/// assert!(matches!(scanner.next(), Some(Ok(Token::Identifier(_)))));
/// assert!(matches!(scanner.next(), Some(Ok(Token::Equal(_)))));
/// assert!(matches!(scanner.next(), Some(Ok(Token::Number(_)))));
/// assert!(matches!(scanner.next(), Some(Ok(Token::Minus(_)))));
/// assert!(matches!(scanner.next(), Some(Ok(Token::Number(_)))));
/// assert!(matches!(scanner.next(), Some(Ok(Token::Semicolon(_)))));
/// assert!(matches!(scanner.next(), None));
/// ```
pub struct Scanner<'c> {
    iter: Peekable<Enumerate<Chars<'c>>>,
    current: Option<(usize, char)>,
    line: usize,
}

impl<'c> From<&'c str> for Scanner<'c> {
    fn from(value: &'c str) -> Self {
        let mut iter = value.chars().enumerate().peekable();
        let current = iter.next();

        Self {
            line: 1,
            iter,
            current,
        }
    }
}

impl<'c> Iterator for Scanner<'c> {
    type Item = Result<Token, ScanError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_whitespace()
            .and_then(|_| self.scan_slash_or_skip_comment())
            .or_else(|| self.scan_identifier())
            .or_else(|| self.scan_number_literal())
            .or_else(|| self.scan_string_literal())
            .or_else(|| self.scan_double_char_token())
            .or_else(|| self.scan_single_char_token())
    }
}

type ScanResult = Option<Result<Token, ScanError>>;

impl<'c> Scanner<'c> {
    fn scan_slash_or_skip_comment(&mut self) -> ScanResult {
        let (i, _) = self.check_current_equals('/')?;

        let next_is_slash = self.advance_if_next_equals('/').is_some();

        if !next_is_slash {
            let token_meta = Span {
                length: 1,
                line: self.line,
                start: i,
            };
            self.advance();
            return Some(Ok(Token::Slash(token_meta)));
        }

        loop {
            let (_, c) = self.advance()?;
            if c == '\n' {
                self.skip_whitespace();
                return None;
            }
        }
    }

    fn skip_whitespace(&mut self) -> Option<()> {
        loop {
            let is_newline = self.check_current_equals('\n').is_some();
            if is_newline {
                self.line += 1;
            }
            let is_whitespace = is_newline
                || self
                    .check_current(|c| matches!(c, ' ' | '\t' | '\r'))
                    .is_some();
            if is_whitespace {
                self.advance()?;
            } else {
                return Some(());
            }
        }
    }

    fn scan_identifier(&mut self) -> ScanResult {
        let (i, c) = self.check_current(char_is_ident_start)?;

        let mut token_meta = Span {
            length: 1,
            line: self.line,
            start: i,
        };

        match c {
            'a' => self.check_keyword(&mut token_meta, "nd", Token::And),
            'c' => self.check_keyword(&mut token_meta, "lass", Token::Class),
            'e' => self.check_keyword(&mut token_meta, "lse", Token::Else),
            'f' => {
                if let Some((_, c)) = self.advance_if_next(char_is_ident_part) {
                    token_meta.length += 1;
                    match c {
                        'a' => self.check_keyword(&mut token_meta, "lse", Token::False),
                        'o' => self.check_keyword(&mut token_meta, "r", Token::For),
                        'u' => self.check_keyword(&mut token_meta, "n", Token::Fun),
                        _ => self.finish_identifier(&mut token_meta),
                    }
                } else {
                    self.advance();
                    Some(Ok(Token::Identifier(token_meta)))
                }
            }
            'i' => self.check_keyword(&mut token_meta, "f", Token::If),
            'n' => self.check_keyword(&mut token_meta, "il", Token::Nil),
            'o' => self.check_keyword(&mut token_meta, "r", Token::Or),
            'p' => self.check_keyword(&mut token_meta, "rint", Token::Print),
            'r' => self.check_keyword(&mut token_meta, "eturn", Token::Return),
            's' => self.check_keyword(&mut token_meta, "uper", Token::Super),
            't' => {
                if let Some((_, c)) = self.advance_if_next(char_is_ident_part) {
                    token_meta.length += 1;
                    match c {
                        'h' => self.check_keyword(&mut token_meta, "is", Token::This),
                        'r' => self.check_keyword(&mut token_meta, "ue", Token::True),
                        _ => self.finish_identifier(&mut token_meta),
                    }
                } else {
                    self.advance();
                    Some(Ok(Token::Identifier(token_meta)))
                }
            }
            'v' => self.check_keyword(&mut token_meta, "ar", Token::Var),
            'w' => self.check_keyword(&mut token_meta, "hile", Token::While),
            _ => self.finish_identifier(&mut token_meta),
        }
    }

    fn check_keyword<F: FnOnce(Span) -> Token>(
        &mut self,
        token_meta: &mut Span,
        rest: &str,
        f: F,
    ) -> ScanResult {
        for c in rest.chars() {
            let next_matches = self.advance_if_next_equals(c).is_some();
            if next_matches {
                token_meta.length += 1;
            } else {
                return self.finish_identifier(token_meta);
            }
        }

        let has_next = self.advance_if_next(char_is_ident_part).is_some();
        if has_next {
            token_meta.length += 1;
            self.finish_identifier(token_meta)
        } else {
            self.advance();
            Some(Ok(f(*token_meta)))
        }
    }

    fn finish_identifier(&mut self, token_meta: &mut Span) -> ScanResult {
        while let Some((_, c)) = self.advance() {
            if !char_is_ident_part(c) {
                break;
            }

            token_meta.length += 1;
        }

        Some(Ok(Token::Identifier(*token_meta)))
    }

    fn scan_number_literal(&mut self) -> ScanResult {
        let (i, _) = self.check_current(char_is_digit)?;

        let mut token_meta = Span {
            length: 1,
            line: self.line,
            start: i,
        };

        let mut is_decimal = false;

        loop {
            let is_digit = self.advance_if_next(char_is_digit).is_some();
            if is_digit {
                token_meta.length += 1;
                continue;
            }

            let is_dot = !is_decimal && self.advance_if_next_equals('.').is_some();

            is_decimal = is_dot && self.advance_if_next(char_is_digit).is_some();

            if is_decimal {
                token_meta.length += 2;
                continue;
            }

            if !is_dot {
                self.advance();
            }
            return Some(Ok(Token::Number(token_meta)));
        }
    }

    fn scan_string_literal(&mut self) -> ScanResult {
        let (i, _) = self.check_current_equals('"')?;

        let mut token_meta = Span {
            length: 1,
            line: self.line,
            start: i,
        };

        while let Some((_, c)) = self.advance() {
            token_meta.length += 1;

            if c == '\n' {
                self.line += 1;
            }
            if c == '"' {
                self.advance();
                return Some(Ok(Token::String(token_meta)));
            }
        }

        Some(Err(ScanError::UnterminatedString(token_meta)))
    }

    fn scan_double_char_token(&mut self) -> ScanResult {
        self.scan_with_equals('!', Token::Bang, Token::BangEqual)
            .or_else(|| self.scan_with_equals('<', Token::Less, Token::LessEqual))
            .or_else(|| self.scan_with_equals('>', Token::Greater, Token::GreaterEqual))
            .or_else(|| self.scan_with_equals('=', Token::Equal, Token::EqualEqual))
    }

    fn scan_with_equals<F1, F2>(&mut self, c: char, f1: F1, f2: F2) -> ScanResult
    where
        F1: FnOnce(Span) -> Token,
        F2: FnOnce(Span) -> Token,
    {
        let (i, _) = self.check_current_equals(c)?;
        let mut token_meta = Span {
            length: 1,
            line: self.line,
            start: i,
        };

        let next_is_equals = self.advance_if_next_equals('=').is_some();

        self.advance();

        if next_is_equals {
            token_meta.length += 1;
            Some(Ok(f2(token_meta)))
        } else {
            Some(Ok(f1(token_meta)))
        }
    }

    fn scan_single_char_token(&mut self) -> ScanResult {
        let result = self.current.map(|(i, c)| {
            let token_meta = Span {
                length: 1,
                line: self.line,
                start: i,
            };
            match c {
                '(' => Ok(Token::LeftParen(token_meta)),
                ')' => Ok(Token::RightParen(token_meta)),
                '{' => Ok(Token::LeftBrace(token_meta)),
                '}' => Ok(Token::RightBrace(token_meta)),
                ';' => Ok(Token::Semicolon(token_meta)),
                ',' => Ok(Token::Comma(token_meta)),
                '.' => Ok(Token::Dot(token_meta)),
                '-' => Ok(Token::Minus(token_meta)),
                '+' => Ok(Token::Plus(token_meta)),
                '*' => Ok(Token::Star(token_meta)),
                '/' => Ok(Token::Slash(token_meta)),
                _ => Err(ScanError::UnrecognisedCharacter(c, token_meta)),
            }
        });

        self.advance();

        result
    }

    fn advance_if_next_equals(&mut self, c: char) -> Option<(usize, char)> {
        let (_, next) = self.iter.peek()?;
        if *next == c {
            self.advance()
        } else {
            None
        }
    }

    fn advance_if_next<F: FnOnce(char) -> bool>(&mut self, f: F) -> Option<(usize, char)> {
        let (_, next) = self.iter.peek()?;
        if f(*next) {
            self.advance()
        } else {
            None
        }
    }

    fn check_current_equals(&self, c: char) -> Option<(usize, char)> {
        self.current
            .and_then(|curr| if curr.1 == c { Some(curr) } else { None })
    }

    fn check_current<F: FnOnce(char) -> bool>(&self, f: F) -> Option<(usize, char)> {
        self.current
            .and_then(|curr| if f(curr.1) { Some(curr) } else { None })
    }

    fn advance(&mut self) -> Option<(usize, char)> {
        self.current = self.iter.next();
        self.current
    }
}

fn char_is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn char_is_ident_part(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn char_is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// Positional data for a token, relative to the input source.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Span {
    /// The first index position of the lexeme.
    pub start: usize,
    /// The total character length of the lexeme.
    pub length: usize,
    /// The line number the lexeme starts on, used for debugging.
    pub line: usize,
}

impl Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.start + 1)
    }
}

impl Default for Span {
    /// Used for testing purposes.
    fn default() -> Self {
        Self {start: 0, length: 0, line: 0}
    }
}

/// Represents a single lexeme of the lox language.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Token {
    // Single-character tokens.
    LeftParen(Span),
    RightParen(Span),
    LeftBrace(Span),
    RightBrace(Span),
    Comma(Span),
    Dot(Span),
    Minus(Span),
    Plus(Span),
    Semicolon(Span),
    Slash(Span),
    Star(Span),
    // One or two character tokens.
    Bang(Span),
    BangEqual(Span),
    Equal(Span),
    EqualEqual(Span),
    Greater(Span),
    GreaterEqual(Span),
    Less(Span),
    LessEqual(Span),
    // Literals.
    Identifier(Span),
    String(Span),
    Number(Span),
    // Keywords.
    And(Span),
    Class(Span),
    Else(Span),
    False(Span),
    For(Span),
    Fun(Span),
    If(Span),
    Nil(Span),
    Or(Span),
    Print(Span),
    Return(Span),
    Super(Span),
    This(Span),
    True(Span),
    Var(Span),
    While(Span),
}

impl Token {
    pub fn meta(&self) -> Span {
        match self {
            Self::LeftParen(t) => *t,
            Self::RightParen(t) => *t,
            Self::LeftBrace(t) => *t,
            Self::RightBrace(t) => *t,
            Self::Comma(t) => *t,
            Self::Dot(t) => *t,
            Self::Minus(t) => *t,
            Self::Plus(t) => *t,
            Self::Semicolon(t) => *t,
            Self::Slash(t) => *t,
            Self::Star(t) => *t,
            Self::Bang(t) => *t,
            Self::BangEqual(t) => *t,
            Self::Equal(t) => *t,
            Self::EqualEqual(t) => *t,
            Self::Greater(t) => *t,
            Self::GreaterEqual(t) => *t,
            Self::Less(t) => *t,
            Self::LessEqual(t) => *t,
            Self::Identifier(t) => *t,
            Self::String(t) => *t,
            Self::Number(t) => *t,
            Self::And(t) => *t,
            Self::Class(t) => *t,
            Self::Else(t) => *t,
            Self::False(t) => *t,
            Self::For(t) => *t,
            Self::Fun(t) => *t,
            Self::If(t) => *t,
            Self::Nil(t) => *t,
            Self::Or(t) => *t,
            Self::Print(t) => *t,
            Self::Return(t) => *t,
            Self::Super(t) => *t,
            Self::This(t) => *t,
            Self::True(t) => *t,
            Self::Var(t) => *t,
            Self::While(t) => *t,
        }
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeftParen(t) => write!(f, "'(' at {t}"),
            Self::RightParen(t) => write!(f, "')' at {t}"),
            Self::LeftBrace(t) => write!(f, "'{{' at {t}"),
            Self::RightBrace(t) => write!(f, "'}}' at {t}"),
            Self::Comma(t) => write!(f, "',' at {t}"),
            Self::Dot(t) => write!(f, "'.' at {t}"),
            Self::Minus(t) => write!(f, "'-' at {t}"),
            Self::Plus(t) => write!(f, "'+' at {t}"),
            Self::Semicolon(t) => write!(f, "';' at {t}"),
            Self::Slash(t) => write!(f, "'/' at {t}"),
            Self::Star(t) => write!(f, "'*' at {t}"),
            Self::Bang(t) => write!(f, "'!' at {t}"),
            Self::BangEqual(t) => write!(f, "'!=' at {t}"),
            Self::Equal(t) => write!(f, "'=' at {t}"),
            Self::EqualEqual(t) => write!(f, "'==' at {t}"),
            Self::Greater(t) => write!(f, "'>' at {t}"),
            Self::GreaterEqual(t) => write!(f, "'>=' at {t}"),
            Self::Less(t) => write!(f, "'<' at {t}"),
            Self::LessEqual(t) => write!(f, "'<=' at {t}"),
            Self::Identifier(t) => write!(f, "<identifier> at {t}"),
            Self::String(t) => write!(f, "<string> at {t}"),
            Self::Number(t) => write!(f, "<number> at {t}"),
            Self::And(t) => write!(f, "'and' at {t}"),
            Self::Class(t) => write!(f, "'class' at {t}"),
            Self::Else(t) => write!(f, "'else' at {t}"),
            Self::False(t) => write!(f, "'false' at {t}"),
            Self::For(t) => write!(f, "'for' at {t}"),
            Self::Fun(t) => write!(f, "'fun' at {t}"),
            Self::If(t) => write!(f, "'if' at {t}"),
            Self::Nil(t) => write!(f, "'nil' at {t}"),
            Self::Or(t) => write!(f, "'or' at {t}"),
            Self::Print(t) => write!(f, "'print' at {t}"),
            Self::Return(t) => write!(f, "'return' at {t}"),
            Self::Super(t) => write!(f, "'super' at {t}"),
            Self::This(t) => write!(f, "'this' at {t}"),
            Self::True(t) => write!(f, "'true' at {t}"),
            Self::Var(t) => write!(f, "'var' at {t}"),
            Self::While(t) => write!(f, "'while' at {t}"),
        }
    }
}

/// All error states that can arise while scanning in input string.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScanError {
    /// Raised when the EOF marker is reached before the closing `"` character of a string.
    UnterminatedString(Span),
    /// Raised when an unrecognised character is encountered outside of a string literal.
    UnrecognisedCharacter(char, Span),
}

impl Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnterminatedString(t) => write!(f, "unterminated string at {t}"),
            Self::UnrecognisedCharacter(c, t) => write!(f, "unrecognised character '{c}' at {t}"),
        }
    }
}

impl Error for ScanError {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn scan_single_character_tokens() {
        let s = "(){};,.-+*/";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::LeftParen(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::RightParen(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::LeftBrace(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::RightBrace(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Semicolon(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Comma(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Dot(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Minus(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Plus(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Star(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Slash(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_yields_error_for_unrecognised_character() {
        let s = "-*@";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Minus(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Star(_))));
        assert_matches!(
            scanner.next(),
            Some(Err(ScanError::UnrecognisedCharacter('@', _)))
        );
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_double_character_tokens() {
        let s = "!!=<<=>>====";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Bang(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::BangEqual(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Less(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::LessEqual(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Greater(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::GreaterEqual(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::EqualEqual(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Equal(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_ignore_whitespace() {
        let s = "! \t\r\n-";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Bang(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Minus(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn newlines_increment_line_count() {
        let s = "!\n!\n!";

        let mut scanner = Scanner::from(s);

        assert_matches!(
            scanner.next(),
            Some(Ok(Token::Bang(Span { line: 1, .. })))
        );
        assert_matches!(
            scanner.next(),
            Some(Ok(Token::Bang(Span { line: 2, .. })))
        );
        assert_matches!(
            scanner.next(),
            Some(Ok(Token::Bang(Span { line: 3, .. })))
        );
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_ignores_comments() {
        let s = "-// a comment\n+";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Minus(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Plus(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_string_literals() {
        let s = "\"hello\"\"multiline\nstring\"";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::String(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::String(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_single_digit_number() {
        let s = "1";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Number(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_multi_digit_number() {
        let s = "112345678901";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Number(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_decimal_number() {
        let s = "1.1";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Number(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_decimal_with_many_leading_digits() {
        let s = "123.1";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Number(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_decimal_with_many_trailing_digits() {
        let s = "3.14567";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Number(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn does_not_scan_decimal_with_no_leading_digit() {
        let s = ".123";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Dot(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Number(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn does_not_scan_decimal_with_no_trailing_digit() {
        let s = "1.";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Number(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Dot(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_identifiers() {
        let s = "a x y thing";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Identifier(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Identifier(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Identifier(_))));
        assert_matches!(scanner.next(), Some(Ok(Token::Identifier(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_and() {
        let s = "and";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::And(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_class() {
        let s = "class";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Class(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_else() {
        let s = "else";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Else(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_false() {
        let s = "false";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::False(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_for() {
        let s = "for";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::For(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_fun() {
        let s = "fun";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Fun(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_if() {
        let s = "if";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::If(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_nil() {
        let s = "nil";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Nil(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_or() {
        let s = "or";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Or(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_print() {
        let s = "print";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Print(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_return() {
        let s = "return";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Return(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_super() {
        let s = "super";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Super(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_this() {
        let s = "this";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::This(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_true() {
        let s = "true";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::True(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_var() {
        let s = "var";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::Var(_))));
        assert_matches!(scanner.next(), None);
    }

    #[test]
    fn scan_keyword_while() {
        let s = "while";

        let mut scanner = Scanner::from(s);

        assert_matches!(scanner.next(), Some(Ok(Token::While(_))));
        assert_matches!(scanner.next(), None);
    }
}
