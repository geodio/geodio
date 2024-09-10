import sys
from antlr4 import *

from core.parser.OperandBuilder import OperandBuilder
from core.parser.tmp.YaguarLexer import YaguarLexer
from core.parser.tmp.YaguarParser import YaguarParser
from antlr4.error.ErrorListener import ErrorListener


class VerboseErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        print(f"Syntax error at {line}:{column} - {msg}")


def parse_input(input_stream):
    lexer = YaguarLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = YaguarParser(stream)

    parser.removeErrorListeners()  # Remove default error listeners
    parser.addErrorListener(
        VerboseErrorListener())  # Add custom error listener

    tree = parser.prog()
    builder = OperandBuilder()
    operand = builder.visit(tree)

    return operand


def main(argv):
    # input_expr = ""
    # input_stream = InputStream(input_expr)
    input_stream = FileStream(argv[1])
    operand = parse_input(input_stream)
    meta_args = {}
    print(operand([3, 4], meta_args))


if __name__ == "__main__":
    main([None, "main.ya"])
