"""
Python AST frontend that parses input into DAG IR
"""
import ast
import inspect
import textwrap
from cutlass_library import DataType
import cutlass
from cutlass.backend.evt.frontend.frontend_base import EVTFrontendBase
from cutlass.backend.epilogue import relu
from cutlass.backend.library import FunctionalOp

class PythonASTFrontend(EVTFrontendBase, ast.NodeVisitor):

    def __init__(self, element_compute=DataType.f32, **kwargs):
        super().__init__(element_compute, **kwargs)
        self.no_imm = False
        self.visiting_return = False

    def parse(self, example_inputs):
        self.example_inputs = example_inputs
        self.source = textwrap.dedent(inspect.getsource(self.__call__))
        self.ast = ast.parse(self.source)
        self.visit(self.ast)

    @staticmethod
    def ast_op_to_bindings(op):
        mapping = {ast.Add: FunctionalOp.Plus, ast.Sub: FunctionalOp.Minus, ast.Mult: FunctionalOp.Multiplies, ast.Div: FunctionalOp.Divides, 'relu': relu.binding_type, 'multiply_add': FunctionalOp.MultiplyAdd, 'sum': (FunctionalOp.Plus, FunctionalOp.AtomicAdd), 'max': (FunctionalOp.Maximum, FunctionalOp.AtomicMaximum)}
        return mapping[op]

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for arg in node.args.args:
            self.visit(arg)
        for expr in node.body:
            self.visit(expr)

    def visit_arg(self, node: ast.arg):
        name = node.arg
        try:
            example_tensor = self.example_inputs[name]
        except:
            raise RuntimeError(f'Example input for {name} is not provided.')
        self.add_load_node(name, example_tensor)

    def visit_Name(self, node: ast.Name):
        return node.id

    def visit_Constant(self, node: ast.Constant):
        if self.no_imm:
            return node.value
        else:
            name = self.add_imm(node.value)
            return name

    def visit_Tuple(self, node: ast.Tuple):
        results = []
        for elt in node.elts:
            results.append(self.visit(elt))
        return tuple(results)

    def visit_keyword(self, node: ast.keyword):
        return {node.arg: self.visit(node.value)}

    def visit_BinOp(self, node: ast.BinOp):
        if self.visiting_return:
            raise SyntaxError('Return value cannot be an expression')
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = self.ast_op_to_bindings(type(node.op))
        name = self.add_compute_node(op)
        self.add_edge(lhs, name, weight=0)
        self.add_edge(rhs, name, weight=1)
        return name

    def visit_Assign(self, node: ast.BinOp):
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        self.add_store_node(target)
        self.add_edge(value, target)
        return target

    def visit_Call(self, node: ast.Call):
        if self.visiting_return:
            raise SyntaxError('Return value cannot be an expression')
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        if func in self.layout_fns.keys():
            self.no_imm = True
            kwargs = {}
            for kw in node.keywords:
                kwargs.update(self.visit(kw))
            self.no_imm = False
            op = self.layout_fns[func]
            name = self.add_layout_node(op, kwargs)
        else:
            op = self.ast_op_to_bindings(func)
            name = self.add_compute_node(op)
        for idx, arg in enumerate(args):
            self.add_edge(arg, name, weight=idx)
        return name

    def visit_Return(self, node: ast.Return):
        self.visiting_return = True
        results = self.visit(node.value)
        self.visiting_return = False
        self.return_names = results
        if not isinstance(results, tuple):
            results = (results,)
        for rst in results:
            try:
                example_tensor = self.example_inputs[rst]
            except:
                raise RuntimeError(f'Example input for {rst} is not provided.')
            self.set_store_tensor(rst, example_tensor)
            self.mark_output(rst)