# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from ..utils import timeout, TimeoutError


logger = getLogger()


def simplify(f, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0
    @timeout(seconds)
    def _simplify(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                logger.warning(f"Detected Dummy symbol when simplifying {f} to {f2}")
                return f
            else:
                return f2
        except TimeoutError:
            return f
        except Exception as e:
            logger.warning(f"{type(e).__name__} exception when simplifying {f}")
            return f
    return _simplify(f)


def count_occurrences(expr):
    """
    Count atom occurrences in an expression.
    """
    if expr.is_Atom:
        return {expr: 1}
    elif expr.is_Add or expr.is_Mul or expr.is_Pow:
        assert len(expr.args) >= 2
        result = {}
        for arg in expr.args:
            sub_count = count_occurrences(arg)
            for k, v in sub_count.items():
                result[k] = result.get(k, 0) + v
        return result
    else:
        assert len(expr.args) == 1, expr
        return count_occurrences(expr.args[0])


def count_occurrences2(expr):
    """
    Count atom occurrences in an expression.
    """
    result = {}
    for sub_expr in sp.preorder_traversal(expr):
        if sub_expr.is_Atom:
            result[sub_expr] = result.get(sub_expr, 0) + 1
    return result


def remove_root_constant_terms(expr, variables, mode):
    """
    Remove root constant terms from a non-constant SymPy expression.
    """
    variables = variables if type(variables) is list else [variables]
    assert mode in ['add', 'mul', 'pow']
    assert any(x in variables for x in expr.free_symbols)
    if mode == 'add' and expr.is_Add or mode == 'mul' and expr.is_Mul:
        args = [arg for arg in expr.args if any(x in variables for x in arg.free_symbols)]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == 'pow' and expr.is_Pow:
        assert len(expr.args) == 2
        if not any(x in variables for x in expr.args[0].free_symbols):
            return expr.args[1]
        elif not any(x in variables for x in expr.args[1].free_symbols):
            return expr.args[0]
        else:
            return expr
    return expr


def remove_mul_const(f, variables):
    """
    Remove the multiplicative factor of an expression, and return it.
    """
    if not f.is_Mul:
        return f, 1
    variables = variables if type(variables) is list else [variables]
    var_args = []
    cst_args = []
    for arg in f.args:
        if any(var in arg.free_symbols for var in variables):
            var_args.append(arg)
        else:
            cst_args.append(arg)
    return sp.Mul(*var_args), sp.Mul(*cst_args)


def extract_non_constant_subtree(expr, variables):
    """
    Extract a non-constant sub-tree from an equation.
    """
    last = expr
    while True:
        last = expr
        expr = remove_root_constant_terms(expr, variables, 'add')
        expr = remove_root_constant_terms(expr, variables, 'mul')
        expr = remove_root_constant_terms(expr, variables, 'pow')
        while len(expr.args) == 1:
            expr = expr.args[0]
        if expr == last:
            return expr


def reindex_coefficients(expr, coefficients):
    """
    Re-index coefficients (i.e. if a1 is there and not a0, replace a1 by a0, and recursively).
    """
    coeffs = sorted([x for x in expr.free_symbols if x in coefficients], key=lambda x: x.name)
    for idx, coeff in enumerate(coefficients):
        if idx >= len(coeffs):
            break
        if coeff != coeffs[idx]:
            expr = expr.subs(coeffs[idx], coeff)
    return expr


def reduce_coefficients(expr, variables, coefficients):
    """
    Reduce coefficients in an expression.
    `sqrt(x)*y*sqrt(1/a0)` -> `a0*sqrt(x)*y`
    `x**(-cos(a0))*y**cos(a0)` -> `x**(-a0)*y**a0`
    """
    temp = sp.Symbol('temp')
    while True:
        last = expr
        for a in coefficients:
            if a not in expr.free_symbols:
                continue
            for subexp in sp.preorder_traversal(expr):
                if a in subexp.free_symbols and not any(var in subexp.free_symbols for var in variables):
                    p = expr.subs(subexp, temp)
                    if a in p.free_symbols:
                        continue
                    else:
                        expr = p.subs(temp, a)
                        break
        if last == expr:
            break
    return expr


def simplify_const_with_coeff(expr, coeff):
    """
    Simplify expressions with constants and coefficients.
    `sqrt(10) * a0 * x` -> `a0 * x`
    `sin(a0 + x + 9/7)` -> `sin(a0 + x)`
    `a0 + x + 9` -> `a0 + x`
    """
    assert coeff.is_Atom
    for parent in sp.preorder_traversal(expr):
        if any(coeff == arg for arg in parent.args):
            break
    if not (parent.is_Add or parent.is_Mul):
        return expr
    removed = [arg for arg in parent.args if len(arg.free_symbols) == 0]
    if len(removed) > 0:
        removed = parent.func(*removed)
        new_coeff = (coeff - removed) if parent.is_Add else (coeff / removed)
        expr = expr.subs(coeff, new_coeff)
    return expr


def simplify_equa_diff(_eq, required=None):
    """
    Simplify a differential equation by removing non-zero factors.
    """
    eq = sp.factor(_eq)
    if not eq.is_Mul:
        return _eq
    args = []
    for arg in eq.args:
        if arg.is_nonzero:
            continue
        if required is None or arg.has(required):
            args.append(arg)
    assert len(args) >= 1
    return args[0] if len(args) == 1 else eq.func(*args)


def smallest_with_symbols(expr, symbols):
    """
    Return the smallest sub-tree in an expression that contains all given symbols.
    """
    assert all(x in expr.free_symbols for x in symbols)
    if len(expr.args) == 1:
        return smallest_with_symbols(expr.args[0], symbols)
    candidates = [arg for arg in expr.args if any(x in arg.free_symbols for x in symbols)]
    return smallest_with_symbols(candidates[0], symbols) if len(candidates) == 1 else expr


def smallest_with(expr, symbol):
    """
    Return the smallest sub-tree in an expression that contains a given symbol.
    """
    assert symbol in expr.free_symbols
    candidates = [arg for arg in expr.args if symbol in arg.free_symbols]
    if len(candidates) > 1 or candidates[0] == symbol:
        return expr
    else:
        return smallest_with(candidates[0], symbol)


def clean_degree2_solution(expr, x, a8, a9):
    """
    Clean solutions of second order differential equations.
    """
    last = expr
    while True:
        for a in [a8, a9]:
            if a not in expr.free_symbols:
                return expr
            small = smallest_with(expr, a)
            if small.is_Add or small.is_Mul:
                counts = count_occurrences2(small)
                if counts[a] == 1 and a in small.args:
                    if x in small.free_symbols:
                        expr = expr.subs(small, small.func(*[arg for arg in small.args if arg == a or x in arg.free_symbols]))
                    else:
                        expr = expr.subs(small, a)
        if expr == last:
            break
        last = expr
    return expr


def has_inf_nan(*args):
    """
    Detect whether some expressions contain a NaN / Infinity symbol.
    """
    for f in args:
        if f.has(sp.nan) or f.has(sp.oo) or f.has(-sp.oo) or f.has(sp.zoo):
            return True
    return False


def has_I(*args):
    """
    Detect whether some expressions contain complex numbers.
    """
    for f in args:
        if f.has(sp.I):
            return True
    return False


if __name__ == '__main__':

    f = sp.Function('f', real=True)
    x = sp.Symbol('x', positive=True, real=True)
    y = sp.Symbol('y', positive=True, real=True)
    z = sp.Symbol('z', positive=True, real=True)
    a0 = sp.Symbol('a0', positive=True, real=True)
    a1 = sp.Symbol('a1', positive=True, real=True)
    a2 = sp.Symbol('a2', positive=True, real=True)
    a3 = sp.Symbol('a3', positive=True, real=True)
    a4 = sp.Symbol('a4', positive=True, real=True)
    a5 = sp.Symbol('a5', positive=True, real=True)
    a6 = sp.Symbol('a6', positive=True, real=True)
    a7 = sp.Symbol('a7', positive=True, real=True)
    a8 = sp.Symbol('a8', positive=True, real=True)
    a9 = sp.Symbol('a9', positive=True, real=True)

    local_dict = {
        'f': f,
        'x': x,
        'y': y,
        'z': z,
        'a0': a0,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'a4': a4,
        'a5': a5,
        'a6': a6,
        'a7': a7,
        'a8': a8,
        'a9': a9,
    }

    failed = 0

    #
    # count occurrences
    #

    def test_count_occurrences(infix, ref_counts):
        expr = parse_expr(infix, local_dict=local_dict)
        counts = count_occurrences(expr)
        if set(counts.keys()) != set(ref_counts.keys()) or not all(ref_counts[k] == v for k, v in counts.items()):
            print(f"Expression {infix} - Expected: {ref_counts} - Returned: {counts})")
            return False
        return True

    def test_count_occurrences2(infix, _counts):
        expr = parse_expr(infix, local_dict=local_dict)
        counts = count_occurrences2(expr)
        assert set(counts.keys()) == set(_counts.keys())
        if not all(_counts[k] == v for k, v in counts.items()):
            print(f"Expression {infix} - Expected: {_counts} - Returned: {counts})")
            return False
        return True

    tests = [
        ('2', {2: 1}),
        ('2*x', {2: 1, x: 1}),
        ('(2*x)**(3*y+1)', {1: 1, 2: 1, 3: 1, x: 1, y: 1}),
        ('(2*x)**(3*y+x+1)', {1: 1, 2: 1, 3: 1, x: 2, y: 1}),
        ('(2*x)**(3*y+x+1)+a0*x', {1: 1, 2: 1, 3: 1, x: 3, y: 1, a0: 1}),
    ]

    for test in tests:
        failed += not test_count_occurrences(*test)
        failed += not test_count_occurrences2(*test)

    #
    # remove root constant terms
    #

    def test_remove_root_constant_terms(infix, ref_output, mode, variables):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = remove_root_constant_terms(expr, variables, mode)
        if output != ref_output:
            print(f"Error when removing constant on expression {infix} with mode {mode} - Expected: {ref_output} - Returned: {output}")
            return False
        return True

    tests = [
        ('x',                     'x',                  'add'),
        ('x + 2',                 'x',                  'add'),
        ('a0*x + 2',              'a0*x',               'add'),
        ('x + exp(2)',            'x',                  'add'),
        ('x + exp(2) * x',        'x + exp(2) * x',     'add'),
        ('x + 2 + a0',            'x',                  'add'),
        ('x + 2 + a0 + z',        'x + z',              'add'),
        ('x + z',                 'x + z',              'add'),
        ('x + 2',                 'x + 2',              'mul'),
        ('x + z',                 'x + z',              'mul'),
        ('x + z',                 'x + z',              'mul'),
        ('a0 * x',                'x',                  'mul'),
        ('(1 / sqrt(a0)) * x',    'x',                  'mul'),
        ('(3 / sqrt(a0)) * x',    'x',                  'mul'),
        ('(3*a0/a1) * sqrt(x)',   'sqrt(x)',            'mul'),
        ('exp(x) / sqrt(a0 + 1)', 'exp(x)',             'mul'),
        ('x + z',                 'x + z',              'mul'),
        ('x + z',                 'x + z',              'mul'),
        ('x + 2',                 'x + 2',              'pow'),
        ('(x + 2) ** 2',          'x + 2',              'pow'),
        ('(x + 2) ** a0',         'x + 2',              'pow'),
        ('(x + 2) ** (a0 + 2)',   'x + 2',              'pow'),
        ('(x + 2) ** (y + 2)',    '(x + 2) ** (y + 2)', 'pow'),
        ('2 ** (x + 2)',          'x + 2',              'pow'),
        ('a0 ** (x + 2)',         'x + 2',              'pow'),
        ('(a0 + 2) ** (x + 2)',   'x + 2',              'pow'),
        ('(y + 2) ** (x + 2)',    '(y + 2) ** (x + 2)', 'pow'),
    ]

    for test in tests:
        failed += not test_remove_root_constant_terms(*test, variables=[x, y, z])

    #
    # extract non-constant sub-tree
    #

    def test_extract_non_constant_subtree(infix, ref_output):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = extract_non_constant_subtree(expr, [x, y, z])
        if output != ref_output:
            print(f"Error when extracting non-constant sub-tree expression {infix} - Expected: {ref_output} - Returned: {output}")
            return False
        return True

    tests = [
        ('x + sqrt(a0 * x)'          , 'x + sqrt(a0 * x)'),
        ('x + sqrt(a0 * x) + 3'      , 'x + sqrt(a0 * x)'),
        ('x + sqrt(a0 * x) + a1'     , 'x + sqrt(a0 * x)'),
        ('x + sqrt(a0 * x) + a0'     , 'x + sqrt(a0 * x) + a0'),
        ('x + sqrt(a0 * x) + 2 * a0' , 'x + sqrt(a0 * x) + 2 * a0'),
        ('a0 * x + x + a0'           , 'a0 * x + x + a0'),
        ('(x + sqrt(a0 * x)) ** 2'   , 'x + sqrt(a0 * x)'),
        ('exp(x + sqrt(a0 * x))'     , 'x + sqrt(a0 * x)'),
        ('exp(x + sqrt(a0 * x))'     , 'x + sqrt(a0 * x)'),
    ]

    for test in tests:
        failed += not test_extract_non_constant_subtree(*test)

    #
    # re-index coefficients
    #

    def test_reindex_coefficients(infix, ref_output):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = reindex_coefficients(expr, [local_dict[f'a{i}'] for i in range(10)])
        if output != ref_output:
            print(f"Error when re-indexing coefficients on expression {infix} - Expected: {ref_output} - Returned: {output}")
            return False
        return True

    tests = [
        ('a0', 'a0'),
        ('a1', 'a0'),
        ('a5', 'a0'),
        ('a9', 'a0'),
        ('a0 + a8', 'a0 + a1'),
        ('a1 + a2', 'a0 + a1'),
        ('a5 + a8', 'a0 + a1'),
        ('a0 * exp(a8)', 'a0 * exp(a1)'),
        ('a4 * exp(a8)', 'a0 * exp(a1)'),
        ('a8 * exp(a4)', 'a1 * exp(a0)'),
        ('(1 + cos(a2)) / ln(a1)', '(1 + cos(a1)) / ln(a0)'),
    ]

    for test in tests:
        failed += not test_reindex_coefficients(*test)

    #
    # reduce coefficients
    #

    def test_reduce_coefficients(infix, ref_output):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = reduce_coefficients(expr, [x, y, z], [local_dict[f'a{i}'] for i in range(10)])
        if output != ref_output:
            print(f"Error when reducing coefficients on expression {infix} - Expected: {ref_output} - Returned: {output}")
            return False
        return True

    tests = [
        ('a0 + 1',                          'a0'),
        ('a0 + x',                          'a0 + x'),
        ('1 / sqrt(a0)',                    'a0'),
        ('1 / (cos(x + sqrt(a0)))',         '1 / (cos(x + a0))'),
        ('a0 / (cos(x + sqrt(a0)))',        'a0 / (cos(x + sqrt(a0)))'),
        ('sqrt(a0) / (cos(x + sqrt(a0)))',  'a0 / (cos(x + a0))'),
        ('ln(a0) / (cos(x + sqrt(a0)))',    'ln(a0 ** 2) / (cos(x + a0))'),
        ('ln(a1) / (cos(x + sqrt(a0)))',    'a1 / (cos(x + a0))'),
        ('sin(a1) * cos(a0 ** 2 + x)',      'a1 * cos(a0 + x)'),
        ('sin(a0) * cos(a0 ** 2 + x)',      'sin(sqrt(a0)) * cos(a0 + x)'),
        ('sin(a0 + x) * cos(a0 ** 2 + x)',  'sin(sqrt(a0) + x) * cos(a0 + x)'),
        ('sin(a0 + x) * cos(a0 ** 2 + a1)', 'sin(a0 + x) * a1'),
        ('sin(a1 + x) * cos(a1 ** 2 + a0)', 'sin(a1 + x) * a0'),
        ('sin(sqrt(a0) + x) * a1',          'sin(a0 + x) * a1')
    ]

    for test in tests:
        failed += not test_reduce_coefficients(*test)

    #
    # simplify constants with coefficients
    #

    def test_simplify_const_with_coeff(infix, ref_output, coeff):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = simplify_const_with_coeff(expr, coeff)
        if output != ref_output:
            print(f"Error when simplifying constants with coefficient {coeff} on expression {infix} - Expected: {ref_output} - Returned: {output}")
            return False
        return True

    tests = [
        ('sqrt(5) * y * x ** (3 / 2) + 5', 'sqrt(5) * y * x ** (3 / 2) + 5', a0),
        ('sqrt(10) * a0 * x', 'a0 * x', a0),
        ('sqrt(10) * a0 * x', 'sqrt(10) * a0 * x', a1),
        ('2 * a0 * x + 1', 'a0 * x + 1', a0),
        ('a0 + tan(x + 5) + 5', 'a0 + tan(x + 5)', a0),
        ('a0 + a1 + 5 + tan(x + 5)', 'a0 + a1 + tan(x + 5)', a0),
        ('a0 + a1 + 5 + tan(x + 5)', 'a0 + a1 + tan(x + 5)', a1),
        ('a0 + x + 9', 'a0 + x', a0),
        ('9 * a0 * x ** 3 + 36 * a0 * x ** 2/5 + x * cos(x)', 'a0 * x ** 3 + 4 * a0 * x ** 2/5 + x * cos(x)', a0),
        ('sqrt(10) * cos((a0 + 1) ** 2) * x', 'sqrt(10) * cos(a0 ** 2) * x', a0),
        ('2 * a0 * x + 1 - 3 * a0 * cos(x)', '(-2 / 3) * a0 * x + 1 + a0 * cos(x)', a0),
        ('ln(sin(a0 + x + 9 / 7) + 1)', 'ln(sin(a0 + x) + 1)', a0),
        ('(a0 + 1) * x ** 2 + x ** 2 + x', 'a0 * x ** 2 + x ** 2 + x', a0),
        ('-3 * a0 - 2 * a0 / x + 3 * x + 2', 'a0 + 2 * a0 / (3 * x) + 3 * x + 2', a0),
    ]

    for test in tests:
        failed += not test_simplify_const_with_coeff(*test)

    # test results
    if failed == 0:
        print("All tests ran successfully.")
    else:
        print(f"{failed} tests failed!")
