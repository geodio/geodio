grammar Yaguar;

tokens { INDENT, DEDENT }

@lexer::header{
from antlr_denter.DenterHelper import DenterHelper
from geodio.core.parser.tmp.YaguarParser import YaguarParser
}
@lexer::members {
class YaguarDenter(DenterHelper):
    def __init__(self, lexer, nl_token, indent_token, dedent_token, ignore_eof):
        super().__init__(nl_token, indent_token, dedent_token, ignore_eof)
        self.lexer: YaguarLexer = lexer

    def pull_token(self):
        return super(YaguarLexer, self.lexer).nextToken()

denter = None

def nextToken(self):
    if not self.denter:
        self.denter = self.YaguarDenter(self, self.NL, YaguarParser.INDENT, YaguarParser.DEDENT, False)
    return self.denter.next_token()

}

NL: ('\r'? '\n' ' '*); // For tabs just switch out ' '* with '\t'*

SPACES : [ \t]+ -> skip ;

COMMENT: '!>' ~[\r\n]* -> skip ;

STRING : '"' ('\\' . | ~["\\])* '"' ;

YES    :
    [Yy] [Ee] [Ss]
    | [Tt] [Rr] [Uu] [Ee]?;
NO     :
    [Nn] [Oo]
    | [Ff] [Aa] [Ll] [Ss] [Ee]?;

OP    :
    'yay';

JUMP: 'jmp';

// Parser rules

prog: ( statement )* ;

statement:
    expr NL                           # ExprStatement
    | ID '=' expr NL                  # AssignmentStatement
    | OP ID '(' params? ')' ':' block # FunctionDeclaration
    | if_statement                    # IfStatement
    | JUMP ID NL                      # MicroJumpStmt
    | ID ':' block                    # LabelDefinition
    ;

if_statement:
    condition+ default? ;

condition:
    expr '?' blockOrExpr;

default:
    '??' blockOrExpr;

blockOrExpr: block | expr NL;

params:
    ID (',' ID)* ;

block:
    INDENT statement+ DEDENT
    ;

//// New rule for defining types and their state
//type_declaration:
//    ID ':state:' INDENT type_body DEDENT;
//
//type_body:
//    (type_member NL)* ;
//
//type_member:
//    ID ':' ID  # TypedMember
//    | ID       # SimpleMember
//    ;
//
//// Method declaration for types
//method_declaration:
//    ID '::' ID '(' params? ')' ':' block;

expr:
    expr op=('*'|'/') expr                     # MulDiv
    | expr op=('+'|'-') expr                   # AddSub
    | expr op=('<'|'>'|'=='|'<='|'>=') expr    # Compare
    | expr op=('&&'|'||') expr                 # Logic
    | expr '^' expr                            # Power
    | expr '>>' NL? INDENT* DEDENT* expr       # ChainExpr
    | '(' expr ')'                             # Parens
    | funcCall                                 # FuncCallExpr
    | '['  exprList? ']'                       # ArrayExpr
    | '{'  exprList? '}'                       # GrpExpr
    | '(' params? ')' '=>' expr                # LambdaExpr
    | ID                                       # Variable
    | NUMBER                                   # Number
    | STRING                                   # String
    | YES                                      # Yes
    | NO                                       # No
    ;

exprList:
    NL? INDENT* DEDENT* expr (',' NL? INDENT* DEDENT* expr)* NL? INDENT* DEDENT*;





    funcCall:
    ID '(' exprList? ')' ;

NUMBER : '-'? [0-9]+ ('.' [0-9]+)? ;
ID     : [a-zA-Z_][a-zA-Z_0-9]* ;



// RULES FOR MICRO


//micro:
//    jumpStmt
//    | label
//    ;
//
//jumpStmt:
//    JUMP ID # MicroJumpStmt  // `jmp label_name` (Jump statement)
//    ;
//
//label:
//    ID ':' block # LabelDefinition  // `label_name:` (Label definition)
//    ;

