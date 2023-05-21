from lark import Lark, Transformer, v_args

math_grammar = """
    ?start: expr
    ?expr: "inv" "(" expr ")" -> inv
         | "suc" "(" expr ")" -> suc
         | "sqr" "(" expr ")" -> sqr
         | "two" "(" expr ")" -> two
         | CNAME -> var
         | expr "-" expr -> sub
         | expr "+" expr -> add
         | expr "*" expr -> mul
         | expr "/" expr -> div
         | "(" expr ")" -> parens
         | "-" expr  -> neg
         | NUMBER -> number
    %import common.CNAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

class MathTransformer(Transformer):
    def inv(self, items):
        return '1/(' + str(items[0]) + ')'
    def suc(self, items):
        return '(' + str(items[0]) + ' + 1)'
    def sqr(self, items):
        return '(' + str(items[0]) + '^2)'
    def two(self, items):
        return '2'
    def var(self, items):
        return str(items[0])
    def sub(self, items):
        return '(' + str(items[0]) + ' - ' + str(items[1]) + ')'
    def add(self, items):
        return '(' + str(items[0]) + ' + ' + str(items[1]) + ')'
    def mul(self, items):
        return '(' + str(items[0]) + ' * ' + str(items[1]) + ')'
    def div(self, items):
        return '(' + str(items[0]) + ' / ' + str(items[1]) + ')'
    def parens(self, items):
        return '(' + str(items[0]) + ')'
    def neg(self, items):
        return '-' + str(items[0])
    def number(self, items):
        return str(items[0])


math_parser = Lark(math_grammar, parser='lalr', transformer=MathTransformer())
parse = math_parser.parse


def translate_to_mathematica(expr):
    return parse(expr)

with open('test.txt', 'r') as file:
    expr = file.read().replace('\n', '')

translated_expr = translate_to_mathematica(expr)


print(translated_expr)