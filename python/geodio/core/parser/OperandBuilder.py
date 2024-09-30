from geodio.core.cell import MetaVariable, MetaArgumented, \
    MetaAssignment, MetaCall, If
from geodio.core.cell.operands import Add, Sub, Prod, Div, Power, Constant, And, Or, \
    GreaterThan, SmallerThan, GreaterThanOrEqual, SmallerThanOrEqual, \
    Seq, Equals, Operand, Label, Jump
from geodio.core.parser.builtins import handle_reserved
from geodio.core.parser import YaguarParser, YaguarListener

reserved = [
    "Linear",
    "Sigmoid",
    "weight",
    "print",
    "train"
]
NULL = Constant(None)


class OperandBuilder(YaguarListener):
    def __init__(self):
        pass

    def visitMicroJumpStmt(self, ctx: YaguarParser.MicroJumpStmtContext):
        label = ctx.ID().getText()
        return Jump(label)

    def visitLabelDefinition(self, ctx: YaguarParser.LabelDefinitionContext):
        label = ctx.ID().getText()
        block = ctx.block()
        children = [self.visit(stmt) for stmt in block.statement()]
        return Label(label, children)



    def visitChainExpr(self, ctx: YaguarParser.ChainExprContext):
        child = ctx.expr(0)
        parent = ctx.expr(1)
        v_child = self.visit(child)
        v_parent = self.visit(parent)
        if isinstance(v_child, YaguarParser.GrpExprContext):
            v_parent.children = v_child([])
        else:
            v_parent.children = [v_child]
        return v_parent

    def visitGrpExpr(self, ctx: YaguarParser.GrpExprContext):
        return Constant([self.visit(expr) for expr in ctx.exprList().expr()])

    def visitIfStatement(self, ctx: YaguarParser.IfStatementContext):
        if_stmt: YaguarParser.If_statementContext = ctx.if_statement()
        prev = None
        for condition in if_stmt.condition():
            condition: YaguarParser.ConditionContext
            cond = self.visit(condition.expr())
            true_case = self.visit(condition.blockOrExpr())
            new = If(cond, [true_case, NULL])
            if prev is not None:
                prev.children[-1] = new
            prev = new

        default = if_stmt.default()
        if default is not None:
            visited_default = self.visit(default.blockOrExpr())
            prev.children[-1] = visited_default
        return prev

    def visitBlockOrExpr(self, ctx: YaguarParser.BlockOrExprContext):
        block = ctx.block()
        if block is None:
            expr = ctx.expr()
            return self.visit(expr)
        children = [self.visit(stmt) for stmt in block.statement()]
        operand = Seq(children)
        return operand

    def visitAssignmentStatement(self,
                                 ctx: YaguarParser.AssignmentStatementContext):
        var_name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        operand = MetaAssignment(var_name, value)
        return operand

    def visitExprStatement(self, ctx: YaguarParser.ExprStatementContext):
        op = self.visit(ctx.expr())
        return op

    def visitFuncCallExpr(self, ctx: YaguarParser.FuncCallExprContext):
        func_name = ctx.funcCall().ID().getText()
        arguments = ctx.funcCall().exprList()
        if arguments:
            args = [self.visit(arg) for arg in arguments.expr()]
        else:
            args = []
        if func_name in reserved:
            x = handle_reserved(func_name, args)(args)
            return x
        operand = MetaCall(func_name, args)
        return operand

    def visitFunctionDeclaration(self,
                                 ctx: YaguarParser.FunctionDeclarationContext):
        func_name = ctx.ID().getText()
        closure = self.get_closure(ctx)
        operand = MetaAssignment(func_name, closure)
        return operand

    def get_closure(self, ctx):
        params = ctx.params()
        if params:
            param_names = [param.getText() for param in ctx.params().ID()]
        else:
            param_names = []
        if hasattr(ctx, 'block'):
            body = ctx.block()
            children = [self.visit(stmt) for stmt in body.statement()]
        else:
            body = ctx.expr()
            children = [self.visit(body)]

        func = Seq(children)
        closure = MetaArgumented(param_names, func)
        return closure

    def visitLambdaExpr(self, ctx: YaguarParser.LambdaExprContext):
        closure = self.get_closure(ctx)
        return closure

    def visitArrayExpr(self, ctx: YaguarParser.ArrayExprContext):
        arguments = ctx.exprList().expr()
        args = []
        if arguments:
            args = [self.visit(arg) for arg in arguments]
        return handle_reserved("weight", args)(args)

    def _invoke_function(self, body, param_names, args):
        # Save current variable state
        saved_meta_args = self.meta_args.copy()
        # Set new variables
        self.meta_args.update(zip(param_names, args))
        # Evaluate the function body
        result = None
        if hasattr(body, 'statement'):
            for stmt in body.statement():
                result = self.visit(stmt)
        else:
            result = self.visit(body)
        # Restore the previous variable state
        self.meta_args = saved_meta_args
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

    def visitYes(self, ctx: YaguarParser.YesContext):
        return Constant(True)

    def visitNo(self, ctx: YaguarParser.NoContext):
        return Constant(False)

    def visitString(self, ctx: YaguarParser.StringContext):
        value = str(ctx.getText()[1:-1])
        return Constant(value)

    def visitVariable(self, ctx: YaguarParser.VariableContext):
        var_name = ctx.getText()
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
