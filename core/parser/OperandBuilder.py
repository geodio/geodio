from core.cell import MetaVariable, PassThrough
from core.parser.builtins import handle_reserved
from core.parser.tmp import YaguarParser, YaguarListener
from core.cell.operands import Add, Sub, Prod, Div, Power, Constant, Variable, \
    And, Or, GreaterThan, SmallerThan, GreaterThanOrEqual, SmallerThanOrEqual, \
    Seq, Equals, Operand, Function, Collector

reserved = [
    "linear",
    "sigmoid",
    "weight"
]


class OperandBuilder(YaguarListener):
    def __init__(self):
        self.variables = {}  # Store variables
        self.functions = {}  # Store function definitions
        self.constants = {}  # Store constants

    def visitAssignmentStatement(self,
                                 ctx: YaguarParser.AssignmentStatementContext):
        var_name = ctx.ID().getText()
        if var_name in self.constants:
            raise ValueError(
                f"Constant {var_name} cannot be re-assigned to a variable")
        value = self.visit(ctx.expr())
        self.variables[var_name] = value
        return value

    def visitExprStatement(self, ctx: YaguarParser.ExprStatementContext):
        op = self.visit(ctx.expr())
        return op

    def visitConstAssignment(self, ctx: YaguarParser.ConstAssignmentContext):
        const_name = ctx.ID().getText()

        if const_name in self.constants:
            raise ValueError(
                f"Constant {const_name} has already been defined.")

        value = self.visit(ctx.expr())
        self.constants[const_name] = value
        return value

    def visitFuncCallExpr(self, ctx: YaguarParser.FuncCallExprContext):
        func_name = ctx.funcCall().ID().getText()
        arguments = ctx.funcCall().args()
        if arguments:
            args = [self.visit(arg) for arg in arguments.expr()]
        else:
            args = []
        if func_name in self.functions:
            return self.functions[func_name](*args)
        if func_name in reserved:
            return handle_reserved(func_name, args)(args)
        if func_name in self.constants:
            func = self.constants[func_name]
            return func(args)
        if func_name in self.variables:
            func = self.variables[func_name]
            return func(args)

        raise ValueError(f"Undefined function {func_name}")

    def visitFunctionDeclaration(self,
                                 ctx: YaguarParser.FunctionDeclarationContext):
        func_name = ctx.ID().getText()
        closure, param_names = self.get_closure(ctx)
        self.functions[func_name] = closure
        operand = Function(0, closure, is_childless=True)
        return operand

    def get_closure(self, ctx):
        params = ctx.params()
        if params:
            param_names = [param.getText() for param in ctx.params().ID()]
        else:
            param_names = []
        frozen_context = (self.variables.copy(), self.functions.copy(),
                          self.constants.copy())
        if hasattr(ctx, 'block'):
            body = ctx.block()
        else:
            body = ctx.expr()
        closure = lambda *args: self._invoke_function(body, param_names,
                                                      args, frozen_context)
        return closure, param_names

    def visitLambdaExpr(self, ctx: YaguarParser.LambdaExprContext):
        closure, param_names = self.get_closure(ctx)
        operand = Function(0, closure, is_childless=True)
        return operand

    def visitArrayExpr(self, ctx: YaguarParser.ArrayExprContext):
        arguments = ctx.expr()
        args = []
        if arguments:
            args = [self.visit(arg) for arg in arguments]
        return handle_reserved("weight", args)(args)

    def _invoke_function(self, body, param_names, args, frozen_context):
        # Save current variable state
        saved_vars = self.variables.copy()
        saved_functions = self.functions.copy()
        saved_constants = self.constants.copy()
        self.variables, self.functions, self.constants = frozen_context
        # Set new variables
        self.variables.update(zip(param_names, args))
        # Evaluate the function body
        result = None
        if hasattr(body, 'statement'):
            for stmt in body.statement():
                result = self.visit(stmt)
        else:
            result = self.visit(body)
        # Restore the previous variable state
        self.variables = saved_vars
        self.functions = saved_functions
        self.constants = saved_constants
        return result

    def visitAddSub(self, ctx: YaguarParser.AddSubContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        if ctx.op.text == '+':
            return Add([left, right], 2)
        else:
            return Sub([left, right])

    def visitMulDiv(self, ctx: YaguarParser.MulDivContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        if ctx.op.text == '*':
            return Prod([left, right])
        else:
            return Div([left, right])

    def visitPower(self, ctx: YaguarParser.PowerContext):
        base = self.visit(ctx.expr(0))
        exponent = self.visit(ctx.expr(1))
        return Power([base, exponent])

    def visitNumber(self, ctx: YaguarParser.NumberContext):
        value = float(ctx.getText())
        return Constant(value)

    def visitVariable(self, ctx: YaguarParser.VariableContext):
        var_name = ctx.getText()
        if var_name in self.variables:
            return self.variables[var_name]
        if var_name in self.constants:
            return self.constants[var_name]
        return MetaVariable(var_name)

    def visitCompare(self, ctx: YaguarParser.CompareContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        if ctx.op.text == '>':
            return GreaterThan([left, right])
        elif ctx.op.text == '<':
            return SmallerThan([left, right])
        elif ctx.op.text == '>=':
            return GreaterThanOrEqual([left, right])
        elif ctx.op.text == '<=':
            return SmallerThanOrEqual([left, right])
        elif ctx.op.text == '==':
            return Equals([left, right])

    def visitLogic(self, ctx: YaguarParser.LogicContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        if ctx.op.text == '&&':
            return And([left, right])
        elif ctx.op.text == '||':
            return Or([left, right])

    def visitParens(self, ctx: YaguarParser.ParensContext):
        return self.visit(ctx.expr())

    def visitProg(self, ctx: YaguarParser.ProgContext):
        return Seq([self.visit(stmt) for stmt in ctx.statement()])

    def visit(self, ctx) -> Operand:
        method_name = 'visit' + ctx.__class__.__name__.replace('Context', '')
        return getattr(self, method_name)(ctx)
