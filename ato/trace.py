import ast


class Canon(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Constant(node.value), node)
        if isinstance(node.value, str):
            return node
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        return node

    def visit_ImportFrom(self, node):
        if node.names:
            node.names = sorted(node.names, key=lambda n: (n.name, n.asname or ''))
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        if all(isinstance(k, ast.Constant) for k in node.keys if k is not None):
            pairs = sorted(zip(node.keys, node.values), key=lambda kv: kv[0].value)
            node.keys, node.values = map(list, zip(*pairs)) if pairs else ([], [])
        return node
