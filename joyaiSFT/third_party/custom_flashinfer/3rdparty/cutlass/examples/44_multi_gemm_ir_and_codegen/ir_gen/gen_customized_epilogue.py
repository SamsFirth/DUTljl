import ast
fuse_gemm_info = [{'epilogue': {'tp': 'LeakyRelu', 'bias': {'addbias': False, 'bias_tp': 'mat'}, 'args': [('float', 'leaky_alpha', 1.3)], 'func': '\ny = max(leaky_alpha * x, x)\ny = y * x\n    '}}]

class AnalysisNodeVisitor(ast.NodeVisitor):

    def visit_Import(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_ImportFrom(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):
        print('Node type: Assign and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_BinOp(self, node):
        print('Node type: BinOp and fields: ', node._fields)
        print('node op: ', type(node.op).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Expr(self, node):
        print('Node type: Expr and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Num(self, node):
        print('Node type: Num and fields: ', node._fields)
        print('Node type: Num: ', node.n)

    def visit_Name(self, node):
        print('Node type: Name and fields: ', node._fields)
        print('Node type: Name and fields: ', type(node.ctx).__name__, node.id)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Str(self, node):
        print('Node type: Str and fields: ', node._fields)

class CodeVisitor(ast.NodeVisitor):

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
            self.generic_visit(node)

    def visit_Assign(self, node):
        print('Assign %s' % node.value)
        self.generic_visit(node)

    def visit_Name(self, node):
        print('Name:', node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        print('Function Name:%s' % node.name.op)
        self.generic_visit(node)
        func_log_stmt = ast.Print(dest=None, values=[ast.Str(s='calling func: %s' % node.name, lineno=0, col_offset=0)], nl=True, lineno=0, col_offset=0)
        node.body.insert(0, func_log_stmt)
visitor = AnalysisNodeVisitor()
code = '\n\na=max(leaky_alpha * x, x +1)\n\n'
visitor.visit(ast.parse(code))