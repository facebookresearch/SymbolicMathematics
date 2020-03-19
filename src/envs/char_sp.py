# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import io
import re
import sys
import math
import itertools
from collections import OrderedDict
import numpy as np
import numexpr as ne
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.cache import clear_cache
from sympy.integrals.risch import NonElementaryIntegral
from sympy.calculus.util import AccumBounds

from ..utils import bool_flag
from ..utils import timeout, TimeoutError
from .sympy_utils import remove_root_constant_terms, reduce_coefficients, reindex_coefficients
from .sympy_utils import extract_non_constant_subtree, simplify_const_with_coeff, simplify_equa_diff, clean_degree2_solution
from .sympy_utils import remove_mul_const, has_inf_nan, has_I, simplify


CLEAR_SYMPY_CACHE_FREQ = 10000


SPECIAL_WORDS = ['<s>', '</s>', '<pad>', '(', ')']
SPECIAL_WORDS = SPECIAL_WORDS + [f'<SPECIAL_{i}>' for i in range(len(SPECIAL_WORDS), 10)]


INTEGRAL_FUNC = {sp.erf, sp.erfc, sp.erfi, sp.erfinv, sp.erfcinv, sp.expint, sp.Ei, sp.li, sp.Li, sp.Si, sp.Ci, sp.Shi, sp.Chi, sp.fresnelc, sp.fresnels}
EXP_OPERATORS = {'exp', 'sinh', 'cosh'}
EVAL_SYMBOLS = {'x', 'y', 'z', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'}
EVAL_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2.1, 3.1]
EVAL_VALUES = EVAL_VALUES + [-x for x in EVAL_VALUES]

TEST_ZERO_VALUES = [0.1, 0.9, 1.1, 1.9]
TEST_ZERO_VALUES = [-x for x in TEST_ZERO_VALUES] + TEST_ZERO_VALUES
ZERO_THRESHOLD = 1e-13


logger = getLogger()


class ValueErrorExpression(Exception):
    pass


class UnknownSymPyOperator(Exception):
    pass


class InvalidPrefixExpression(Exception):

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


def count_nested_exp(s):
    """
    Return the maximum number of nested exponential functions in an infix expression.
    """
    stack = []
    count = 0
    max_count = 0
    for v in re.findall('[+-/*//()]|[a-zA-Z0-9]+', s):
        if v == '(':
            stack.append(v)
        elif v == ')':
            while True:
                x = stack.pop()
                if x in EXP_OPERATORS:
                    count -= 1
                if x == '(':
                    break
        else:
            stack.append(v)
            if v in EXP_OPERATORS:
                count += 1
                max_count = max(max_count, count)
    assert len(stack) == 0
    return max_count


def is_valid_expr(s):
    """
    Check that we are able to evaluate an expression (and that it will not blow in SymPy evaluation).
    """
    s = s.replace('Derivative(f(x),x)', '1')
    s = s.replace('Derivative(1,x)', '1')
    s = s.replace('(E)', '(exp(1))')
    s = s.replace('(I)', '(1)')
    s = s.replace('(pi)', '(1)')
    s = re.sub(r'(?<![a-z])(f|g|h|Abs|sign|ln|sin|cos|tan|sec|csc|cot|asin|acos|atan|asec|acsc|acot|tanh|sech|csch|coth|asinh|acosh|atanh|asech|acoth|acsch)\(', '(', s)
    count = count_nested_exp(s)
    if count >= 4:
        return False
    for v in EVAL_VALUES:
        try:
            local_dict = {s: (v + 1e-4 * i) for i, s in enumerate(EVAL_SYMBOLS)}
            value = ne.evaluate(s, local_dict=local_dict).item()
            if not (math.isnan(value) or math.isinf(value)):
                return True
        except (FloatingPointError, ZeroDivisionError, TypeError, MemoryError):
            continue
    return False


def eval_test_zero(eq):
    """
    Evaluate an equation by replacing all its free symbols with random values.
    """
    variables = eq.free_symbols
    assert len(variables) <= 3
    outputs = []
    for values in itertools.product(*[TEST_ZERO_VALUES for _ in range(len(variables))]):
        _eq = eq.subs(zip(variables, values)).doit()
        outputs.append(float(sp.Abs(_eq.evalf())))
    return outputs


class CharSPEnvironment(object):

    TRAINING_TASKS = {'prim_fwd', 'prim_bwd', 'prim_ibp', 'ode1', 'ode2'}

    # https://docs.sympy.org/latest/modules/functions/elementary.html#real-root

    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',
        sp.exp: 'exp',
        sp.log: 'ln',
        sp.Abs: 'abs',
        sp.sign: 'sign',
        # Trigonometric Functions
        sp.sin: 'sin',
        sp.cos: 'cos',
        sp.tan: 'tan',
        sp.cot: 'cot',
        sp.sec: 'sec',
        sp.csc: 'csc',
        # Trigonometric Inverses
        sp.asin: 'asin',
        sp.acos: 'acos',
        sp.atan: 'atan',
        sp.acot: 'acot',
        sp.asec: 'asec',
        sp.acsc: 'acsc',
        # Hyperbolic Functions
        sp.sinh: 'sinh',
        sp.cosh: 'cosh',
        sp.tanh: 'tanh',
        sp.coth: 'coth',
        sp.sech: 'sech',
        sp.csch: 'csch',
        # Hyperbolic Inverses
        sp.asinh: 'asinh',
        sp.acosh: 'acosh',
        sp.atanh: 'atanh',
        sp.acoth: 'acoth',
        sp.asech: 'asech',
        sp.acsch: 'acsch',
        # Derivative
        sp.Derivative: 'derivative',
    }

    OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'rac': 2,
        'inv': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'pow5': 1,
        'sqrt': 1,
        'exp': 1,
        'ln': 1,
        'abs': 1,
        'sign': 1,
        # Trigonometric Functions
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'cot': 1,
        'sec': 1,
        'csc': 1,
        # Trigonometric Inverses
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'acot': 1,
        'asec': 1,
        'acsc': 1,
        # Hyperbolic Functions
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'coth': 1,
        'sech': 1,
        'csch': 1,
        # Hyperbolic Inverses
        'asinh': 1,
        'acosh': 1,
        'atanh': 1,
        'acoth': 1,
        'asech': 1,
        'acsch': 1,
        # Derivative
        'derivative': 2,
        # custom functions
        'f': 1,
        'g': 2,
        'h': 3,
    }

    def __init__(self, params):

        self.max_int = params.max_int
        self.max_ops = params.max_ops
        self.max_ops_G = params.max_ops_G
        self.int_base = params.int_base
        self.balanced = params.balanced
        self.positive = params.positive
        self.precision = params.precision
        self.n_variables = params.n_variables
        self.n_coefficients = params.n_coefficients
        self.max_len = params.max_len
        self.clean_prefix_expr = params.clean_prefix_expr
        assert self.max_int >= 1
        assert abs(self.int_base) >= 2
        assert self.precision >= 2

        # parse operators with their weights
        self.operators = sorted(list(self.OPERATORS.keys()))
        ops = params.operators.split(',')
        ops = sorted([x.split(':') for x in ops])
        assert len(ops) >= 1 and all(o in self.OPERATORS for o, _ in ops)
        self.all_ops = [o for o, _ in ops]
        self.una_ops = [o for o, _ in ops if self.OPERATORS[o] == 1]
        self.bin_ops = [o for o, _ in ops if self.OPERATORS[o] == 2]
        logger.info(f"Unary operators: {self.una_ops}")
        logger.info(f"Binary operators: {self.bin_ops}")
        self.all_ops_probs = np.array([float(w) for _, w in ops]).astype(np.float64)
        self.una_ops_probs = np.array([float(w) for o, w in ops if self.OPERATORS[o] == 1]).astype(np.float64)
        self.bin_ops_probs = np.array([float(w) for o, w in ops if self.OPERATORS[o] == 2]).astype(np.float64)
        self.all_ops_probs = self.all_ops_probs / self.all_ops_probs.sum()
        self.una_ops_probs = self.una_ops_probs / self.una_ops_probs.sum()
        self.bin_ops_probs = self.bin_ops_probs / self.bin_ops_probs.sum()

        assert len(self.all_ops) == len(set(self.all_ops)) >= 1
        assert set(self.all_ops).issubset(set(self.operators))
        assert len(self.all_ops) == len(self.una_ops) + len(self.bin_ops)

        # symbols / elements
        self.constants = ['pi', 'E']
        self.variables = OrderedDict({
            'x': sp.Symbol('x', real=True, nonzero=True),  # , positive=True
            'y': sp.Symbol('y', real=True, nonzero=True),  # , positive=True
            'z': sp.Symbol('z', real=True, nonzero=True),  # , positive=True
            't': sp.Symbol('t', real=True, nonzero=True),  # , positive=True
        })
        self.coefficients = OrderedDict({
            f'a{i}': sp.Symbol(f'a{i}', real=True)
            for i in range(10)
        })
        self.functions = OrderedDict({
            'f': sp.Function('f', real=True, nonzero=True),
            'g': sp.Function('g', real=True, nonzero=True),
            'h': sp.Function('h', real=True, nonzero=True),
        })
        self.symbols = ['I', 'INT+', 'INT-', 'INT', 'FLOAT', '-', '.', '10^', 'Y', "Y'", "Y''"]
        if self.balanced:
            assert self.int_base > 2
            max_digit = (self.int_base + 1) // 2
            self.elements = [str(i) for i in range(max_digit - abs(self.int_base), max_digit)]
        else:
            self.elements = [str(i) for i in range(abs(self.int_base))]
        assert 1 <= self.n_variables <= len(self.variables)
        assert 0 <= self.n_coefficients <= len(self.coefficients)
        assert all(k in self.OPERATORS for k in self.functions.keys())
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # SymPy elements
        self.local_dict = {}
        for k, v in list(self.variables.items()) + list(self.coefficients.items()) + list(self.functions.items()):
            assert k not in self.local_dict
            self.local_dict[k] = v

        # vocabulary
        self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) + list(self.coefficients.keys()) + self.operators + self.symbols + self.elements
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"words: {self.word2id}")

        # leaf probabilities
        s = [float(x) for x in params.leaf_probs.split(',')]
        assert len(s) == 4 and all(x >= 0 for x in s)
        self.leaf_probs = np.array(s).astype(np.float64)
        self.leaf_probs = self.leaf_probs / self.leaf_probs.sum()
        assert self.leaf_probs[0] > 0
        assert (self.leaf_probs[1] == 0) == (self.n_coefficients == 0)

        # possible leaves
        self.n_leaves = self.n_variables + self.n_coefficients
        if self.leaf_probs[2] > 0:
            self.n_leaves += self.max_int * (1 if self.positive else 2)
        if self.leaf_probs[3] > 0:
            self.n_leaves += len(self.constants)
        logger.info(f"{self.n_leaves} possible leaves.")

        # generation parameters
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)

        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist(params.max_ops)
        self.ubi_dist = self.generate_ubi_dist(params.max_ops)

        # rewrite expressions
        self.rewrite_functions = [x for x in params.rewrite_functions.split(',') if x != '']
        assert len(self.rewrite_functions) == len(set(self.rewrite_functions))
        assert all(x in ['expand', 'factor', 'expand_log', 'logcombine', 'powsimp', 'simplify'] for x in self.rewrite_functions)

        # valid check
        logger.info(f"Checking expressions in {str(EVAL_VALUES)}")

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def generate_bin_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # initialize Catalan numbers
        catalans = [1]
        for i in range(1, 2 * max_ops + 1):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

        # enumerate possible trees
        D = []
        for e in range(max_ops + 1):  # number of empty nodes
            s = []
            for n in range(2 * max_ops - e + 1):  # number of operators
                if e == 0:
                    s.append(0)
                elif e == 1:
                    s.append(catalans[n])
                else:
                    s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
            D.append(s)
        return D

    def generate_ubi_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the tranposed version of D, then transpose it
        D = []
        D.append([0] + ([self.nl ** i for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(self.nl * s[e - 1] + self.p1 * D[n - 1][e] + self.p2 * D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
        return D

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        """
        base = self.int_base
        balanced = self.balanced
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        if base < 0 or balanced:
            res.append('INT')
        else:
            res.append('INT-' if neg else 'INT+')
        return res[::-1]

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.int_base
        balanced = self.balanced
        val = 0
        if not (balanced and lst[0] == 'INT' or base >= 2 and lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
            raise InvalidPrefixExpression(f"Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == 'INT-':
            val = -val
        return val, i + 1

    def sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1])
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1])
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def get_leaf(self, max_int, rng):
        """
        Generate a leaf.
        """
        self.leaf_probs
        leaf_type = rng.choice(4, p=self.leaf_probs)
        if leaf_type == 0:
            return [list(self.variables.keys())[rng.randint(self.n_variables)]]
        elif leaf_type == 1:
            return [list(self.coefficients.keys())[rng.randint(self.n_coefficients)]]
        elif leaf_type == 2:
            c = rng.randint(1, max_int + 1)
            c = c if (self.positive or rng.randint(2) == 0) else -c
            return self.write_int(c)
        else:
            return [self.constants[rng.randint(len(self.constants))]]

    def _generate_expr(self, nb_total_ops, max_int, rng, require_x=False, require_y=False, require_z=False):
        """
        Create a tree with exactly `nb_total_ops` operators.
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops, rng)
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            nb_empty += self.OPERATORS[op] - 1 - skipped  # created empty nodes - skipped future leaves
            t_leaves += self.OPERATORS[op] - 1            # update number of total leaves
            l_leaves += skipped                           # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = stack[:pos] + [op] + [None for _ in range(self.OPERATORS[op])] + stack[pos + 1:]

        # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves

        # create leaves
        # optionally add variables x, y, z if possible
        assert not require_z or require_y
        assert not require_y or require_x
        leaves = [self.get_leaf(max_int, rng) for _ in range(t_leaves)]
        if require_z and t_leaves >= 2:
            leaves[1] = ['z']
        if require_y:
            leaves[0] = ['y']
        if require_x and not any(len(leaf) == 1 and leaf[0] == 'x' for leaf in leaves):
            leaves[-1] = ['x']
        rng.shuffle(leaves)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1:]
        assert len(leaves) == 0

        return stack

    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'pow':
            return f'({args[0]})**({args[1]})'
        elif token == 'rac':
            return f'({args[0]})**(1/({args[1]}))'
        elif token == 'abs':
            return f'Abs({args[0]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token == 'pow2':
            return f'({args[0]})**2'
        elif token == 'pow3':
            return f'({args[0]})**3'
        elif token == 'pow4':
            return f'({args[0]})**4'
        elif token == 'pow5':
            return f'({args[0]})**5'
        elif token in ['sign', 'sqrt', 'exp', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']:
            return f'{token}({args[0]})'
        elif token == 'derivative':
            return f'Derivative({args[0]},{args[1]})'
        elif token == 'f':
            return f'f({args[0]})'
        elif token == 'g':
            return f'g({args[0]},{args[1]})'
        elif token == 'h':
            return f'h({args[0]},{args[1]},{args[2]})'
        elif token.startswith('INT'):
            return f'{token[-1]}{args[0]}'
        else:
            return token
        raise InvalidPrefixExpression(f"Unknown token in prefix expression: {token}, with arguments {args}")

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        elif t in self.variables or t in self.coefficients or t in self.constants or t == 'I':
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]

    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'

    def rewrite_sympy_expr(self, expr):
        """
        Rewrite a SymPy expression.
        """
        expr_rw = expr
        for f in self.rewrite_functions:
            if f == 'expand':
                expr_rw = sp.expand(expr_rw)
            elif f == 'factor':
                expr_rw = sp.factor(expr_rw)
            elif f == 'expand_log':
                expr_rw = sp.expand_log(expr_rw, force=True)
            elif f == 'logcombine':
                expr_rw = sp.logcombine(expr_rw, force=True)
            elif f == 'powsimp':
                expr_rw = sp.powsimp(expr_rw, force=True)
            elif f == 'simplify':
                expr_rw = simplify(expr_rw, seconds=1)
        return expr_rw

    def infix_to_sympy(self, infix, no_rewrite=False):
        """
        Convert an infix expression to SymPy.
        """
        if not is_valid_expr(infix):
            raise ValueErrorExpression
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueErrorExpression
        if not no_rewrite:
            expr = self.rewrite_sympy_expr(expr)
        return expr

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # derivative operator
        if op == 'derivative':
            assert n_args >= 2
            assert all(len(arg) == 2 and str(arg[0]) in self.variables and int(arg[1]) >= 1 for arg in expr.args[1:]), expr.args
            parse_list = self.sympy_to_prefix(expr.args[0])
            for var, degree in expr.args[1:]:
                parse_list = ['derivative' for _ in range(int(degree))] + parse_list + [str(var) for _ in range(int(degree))]
            return parse_list

        assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)

        # square root
        if op == 'pow' and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
            return ['sqrt'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ['div'] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ['E']
        elif expr == sp.pi:
            return ['pi']
        elif expr == sp.I:
            return ['I']
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # environment function
        for func_name, func in self.functions.items():
            if isinstance(expr, func):
                return self._sympy_to_prefix(func_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def reduce_coefficients(self, expr):
        return reduce_coefficients(expr, self.variables.values(), self.coefficients.values())

    def reindex_coefficients(self, expr):
        if self.n_coefficients == 0:
            return expr
        return reindex_coefficients(expr, list(self.coefficients.values())[:self.n_coefficients])

    def extract_non_constant_subtree(self, expr):
        return extract_non_constant_subtree(expr, self.variables.values())

    def simplify_const_with_coeff(self, expr, coeffs=None):
        if coeffs is None:
            coeffs = self.coefficients.values()
        for coeff in coeffs:
            expr = simplify_const_with_coeff(expr, coeff)
        return expr

    def clean_prefix(self, expr):
        """
        Clean prefix expressions before they are converted to PyTorch.
        "f x" -> "Y"
        "derivative f x x" -> "Y'"
        """
        if not self.clean_prefix_expr:
            return expr
        expr = " ".join(expr)
        expr = expr.replace("f x", "Y")
        expr = expr.replace("derivative Y x", "Y'")
        expr = expr.replace("derivative Y' x", "Y''")
        expr = expr.split()
        return expr

    def unclean_prefix(self, expr):
        """
        Unclean prefix expressions before they are converted to PyTorch.
        "Y" -> "f x"
        "Y'" -> "derivative f x x"
        """
        if not self.clean_prefix_expr:
            return expr
        expr = " ".join(expr)
        expr = expr.replace("Y''", "derivative Y' x")
        expr = expr.replace("Y'", "derivative Y x")
        expr = expr.replace("Y", "f x")
        expr = expr.split()
        return expr

    @timeout(3)
    def gen_prim_fwd(self, rng):
        """
        Generate pairs of (function, primitive).
        Start by generating a random function f, and use SymPy to compute F.
        """
        x = self.variables['x']
        if rng.randint(40) == 0:
            nb_ops = rng.randint(0, 3)
        else:
            nb_ops = rng.randint(3, self.max_ops + 1)

        if not hasattr(self, 'prim_stats'):
            self.prim_stats = np.zeros(10, dtype=np.int64)

        try:
            # generate an expression and rewrite it,
            # avoid issues in 0 and convert to SymPy
            f_expr = self._generate_expr(nb_ops, self.max_int, rng)
            infix = self.prefix_to_infix(f_expr)
            f = self.infix_to_sympy(infix)

            # skip constant expressions
            if x not in f.free_symbols:
                return None

            # remove additive constant, re-index coefficients
            if rng.randint(2) == 0:
                f = remove_root_constant_terms(f, x, 'add')
            f = self.reduce_coefficients(f)
            f = self.simplify_const_with_coeff(f)
            f = self.reindex_coefficients(f)

            # compute its primitive, and rewrite it
            self.prim_stats[-1] += 1
            F = sp.integrate(f, x, risch=True)
            if isinstance(F, NonElementaryIntegral):
                self.prim_stats[0] += 1
                return None
            F = F.doit()
            if has_inf_nan(F) or isinstance(F, NonElementaryIntegral) or F.has(sp.Integral) or F.has(sp.Piecewise):
                self.prim_stats[1] += 1
                return None
            if any(op.func in INTEGRAL_FUNC for op in sp.preorder_traversal(F)):
                self.prim_stats[2] += 1
                return None
            self.prim_stats[3] += 1

            # skip invalid expressions
            if has_inf_nan(f, F):
                return None

            # convert back to prefix
            f_prefix = self.sympy_to_prefix(f)
            F_prefix = self.sympy_to_prefix(F)

            # skip too long sequences
            if max(len(f_prefix), len(F_prefix)) + 2 > self.max_len:
                return None

            # skip when the number of operators is too far from expected
            real_nb_ops = sum(1 if op in self.OPERATORS else 0 for op in f_prefix)
            if real_nb_ops < nb_ops / 2:
                return None

            self.prim_stats[4] += 1
            if self.prim_stats[-1] % 500 == 0:
                logger.debug(f"{self.worker_id:>2} PRIM STATS {self.prim_stats}")

        except TimeoutError:
            raise
        except (ValueError, AttributeError, TypeError, OverflowError, NotImplementedError, UnknownSymPyOperator, ValueErrorExpression):
            return None
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, infix, e.args))
            return None

        # define input / output
        x = ['sub', 'derivative', 'f', 'x', 'x'] + f_prefix
        y = F_prefix
        x = self.clean_prefix(x)
        y = self.clean_prefix(y)

        return x, y

    @timeout(5)
    def gen_prim_bwd(self, rng, predict_primitive):
        """
        Generate pairs of (function, derivative) or (function, primitive).
        Start by generating a random function f, and use SymPy to compute f'.
        """
        x = self.variables['x']
        if rng.randint(40) == 0:
            nb_ops = rng.randint(0, 4)
        else:
            nb_ops = rng.randint(4, self.max_ops + 1)

        try:
            # generate an expression and rewrite it,
            # avoid issues in 0 and convert to SymPy
            F_expr = self._generate_expr(nb_ops, self.max_int, rng)
            infix = self.prefix_to_infix(F_expr)
            F = self.infix_to_sympy(infix)

            # skip constant expressions
            if x not in F.free_symbols:
                return None

            # remove additive constant, re-index coefficients
            F = remove_root_constant_terms(F, x, 'add')
            F = self.reduce_coefficients(F)
            F = self.simplify_const_with_coeff(F)
            F = self.reindex_coefficients(F)

            # compute the derivative, and simplify it
            f = sp.diff(F, x)
            if rng.randint(2) == 1:
                f = simplify(f, seconds=2)

            # skip invalid expressions
            if has_inf_nan(f, F):
                return None

            # convert back to prefix
            f_prefix = self.sympy_to_prefix(f)
            F_prefix = self.sympy_to_prefix(F)

            # skip too long sequences
            if max(len(f_prefix), len(F_prefix)) + 2 > self.max_len:
                return None

            # skip when the number of operators is too far from expected
            real_nb_ops = sum(1 if op in self.OPERATORS else 0 for op in F_prefix)
            if real_nb_ops < nb_ops / 2:
                return None

        except TimeoutError:
            raise
        except (ValueErrorExpression, UnknownSymPyOperator, OverflowError, TypeError):
            return None
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, infix, e.args))
            return None

        # define input / output
        if predict_primitive:
            x = ['sub', 'derivative', 'f', 'x', 'x'] + f_prefix
            y = F_prefix
        else:
            x = ['sub', 'f', 'x', 'derivative'] + F_prefix + ['x']
            y = f_prefix
        x = self.clean_prefix(x)
        y = self.clean_prefix(y)

        return x, y

    PRIM_CACHE = {}
    PRIM_COUNT = [0, 0]

    @timeout(8)
    def gen_prim_ibp(self, rng):
        """
        Generate pairs of (function, primitive).
        Start by generating random functions F and G, and use the fact that:
            (FG)(x) = (FG)(0) + int(F*g, x=0..x) + int(g*G, x=0..x)
        where f = F' ang g = G'.
        """
        def update_cache(f, F):
            if x not in f.free_symbols:
                return
            f, c = remove_mul_const(f, x)
            F = remove_root_constant_terms(F, x, 'add') / c
            prev_F = self.PRIM_CACHE.get(f)
            if prev_F is None or sp.count_ops(F) < sp.count_ops(prev_F):
                self.PRIM_CACHE[f] = F

        def read_cache(f):
            if x not in f.free_symbols:
                return f * x
            f, c = remove_mul_const(f, x)
            F = self.PRIM_CACHE.get(f)
            return None if F is None else F * c

        x = self.variables['x']
        if rng.randint(40) == 0:
            nb_ops_F = rng.randint(0, 4)
        else:
            nb_ops_F = rng.randint(4, self.max_ops + 1)
        nb_ops_G = rng.randint(1, self.max_ops_G + 1)

        try:
            # generate an expression and rewrite it,
            # avoid issues in 0 and convert to SymPy
            F_expr = self._generate_expr(nb_ops_F, self.max_int, rng)
            G_expr = self._generate_expr(nb_ops_G, self.max_int, rng)
            F_infix = self.prefix_to_infix(F_expr)
            G_infix = self.prefix_to_infix(G_expr)
            F = self.infix_to_sympy(F_infix)
            G = self.infix_to_sympy(G_infix)

            # skip constant expressions
            if x not in F.free_symbols or x not in G.free_symbols:
                return None

            # remove additive constant, re-index coefficients
            if rng.randint(2) == 0:
                F = remove_root_constant_terms(F, x, 'add')
            if rng.randint(2) == 0:
                G = remove_root_constant_terms(G, x, 'add')
            F = self.reduce_coefficients(F)
            G = self.reduce_coefficients(G)
            F = self.simplify_const_with_coeff(F)
            G = self.simplify_const_with_coeff(G)
            F = self.reindex_coefficients(F)
            G = self.reindex_coefficients(G)

            # compute derivatives, and simplify it
            f = sp.diff(F, x)
            g = sp.diff(G, x)
            f = simplify(f, seconds=1)
            g = simplify(g, seconds=1)

            # remove multiplicative constant in g and G
            g, g_const = remove_mul_const(g, x)
            G = G / g_const

            # skip invalid expressions
            if has_inf_nan(f, F, g, G):
                return None

            # update cache
            update_cache(f, F)
            update_cache(g, G)

            # search for primitives
            self.PRIM_COUNT[1] += 1
            fG = simplify(f * G, seconds=1)
            fG_prim = read_cache(fG)
            if fG_prim is not None:
                h = F * g
                H = F * G - fG_prim
            else:
                Fg = simplify(F * g, seconds=1)
                Fg_prim = read_cache(Fg)
                if Fg_prim is not None:
                    h = f * G
                    H = F * G - Fg_prim
                else:
                    return None

            # log match accuracy
            self.PRIM_COUNT[0] += 1
            if self.PRIM_COUNT[1] % 100 == 0:
                logger.info(f"PRIM_COUNT {self.PRIM_COUNT[0]} / {self.PRIM_COUNT[1]} = {100 * self.PRIM_COUNT[0] / self.PRIM_COUNT[1]}%")

            # simplify
            H = remove_root_constant_terms(H, x, 'add')
            H = simplify(H, seconds=1)
            h = simplify(h, seconds=1)

            # skip invalid expressions
            if has_inf_nan(h, H):
                return None

            # skip constant expressions
            if x not in h.free_symbols:
                return None

            # remove multiplicative constant in h and H
            if rng.randint(2) == 0:
                h, h_const = remove_mul_const(h, x)
                H = H / h_const

            # update cache
            update_cache(h, H)

            # convert back to prefix
            h_prefix = self.sympy_to_prefix(h)
            H_prefix = self.sympy_to_prefix(H)

            # skip too long sequences
            if max(len(h_prefix), len(H_prefix)) + 2 > self.max_len:
                return None

        except TimeoutError:
            raise
        except (ValueErrorExpression, UnknownSymPyOperator, OverflowError, TypeError):
            return None
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, F_infix, e.args))
            return None

        # define input / output
        x = ['sub', 'derivative', 'f', 'x', 'x'] + h_prefix
        y = H_prefix
        x = self.clean_prefix(x)
        y = self.clean_prefix(y)

        return x, y

    @timeout(8)
    def gen_ode1(self, rng):
        """
        Generate first order differential equations.
        """
        assert self.n_coefficients <= 8
        x = self.variables['x']
        y = self.variables['y']
        f = self.functions['f']
        a8 = self.coefficients['a8']

        nb_ops = rng.randint(3, self.max_ops + 1)

        try:
            # generate an expression and rewrite it,
            # avoid issues in 0 and convert to SymPy
            # here, y has the role of a constant
            expr = self._generate_expr(nb_ops, self.max_int, rng, require_x=True, require_y=True)
            infix = self.prefix_to_infix(expr)
            expr = self.infix_to_sympy(infix).subs(y, a8)

            # skip constant expressions
            if x not in expr.free_symbols or a8 not in expr.free_symbols:
                return None

            # reduce expression coefficients
            # simplify constants with coefficients
            # re-index coefficients
            expr = self.reduce_coefficients(expr)
            expr = self.simplify_const_with_coeff(expr, coeffs=[a8])
            expr = self.reindex_coefficients(expr)

            # skip invalid expressions
            if has_inf_nan(expr) or has_I(expr):
                return None

            # convert the expression to prefix
            expr_prefix = self.sympy_to_prefix(expr)

            # skip too long expressions
            if len(expr_prefix) + 2 > self.max_len:
                return None

            # skip when the number of operators in f is too far from expected
            real_nb_ops = sum(1 if op in self.OPERATORS else 0 for op in expr_prefix)
            if real_nb_ops < nb_ops / 2:
                return None

            # express the constant a8 in terms of x and f
            solve_a8 = sp.solve(f(x) - expr, a8, check=False, simplify=False)
            if len(solve_a8) == 0 or type(solve_a8) is not list:
                return None
            solve_a8 = [s for s in solve_a8 if x in s.free_symbols]
            if len(solve_a8) == 0:
                return None
            _a8 = solve_a8[rng.randint(len(solve_a8))]
            if type(_a8) is tuple or type(_a8) is sp.Piecewise:
                return None

            # compute differential equation
            # simplify the differential equation by removing positive terms
            eq = _a8.diff(x)
            if not eq.has(f(x).diff(x)):
                return None
            eq = simplify_equa_diff(eq, required=f(x).diff(x))
            if rng.randint(2) == 1:
                eq = simplify(eq, seconds=1)

            # skip invalid expressions
            if has_inf_nan(eq) or has_I(eq):
                return None

            # convert equation to prefix
            eq_prefix = self.sympy_to_prefix(eq)
            eq_prefix = self.clean_prefix(eq_prefix)

            # skip too long equations
            if len(eq_prefix) + 2 > self.max_len:
                return None

            # perform checks
            check_eq = simplify(eq.subs(f(x), expr).doit(), seconds=1)
            if check_eq != 0:
                eval_values = eval_test_zero(check_eq)
                if len([y for y in eval_values if y <= ZERO_THRESHOLD]) / len(eval_values) < 0.2:
                    return None

        except TimeoutError:
            raise
        except (ValueError, NotImplementedError, AttributeError, RecursionError, ValueErrorExpression, UnknownSymPyOperator):
            return None
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, infix, e.args))
            return None

        return eq_prefix, expr_prefix

    @timeout(8)
    def gen_ode2(self, rng):
        """
        Generate second order differential equations.
        """
        assert self.n_coefficients <= 8
        x = self.variables['x']
        y = self.variables['y']
        z = self.variables['z']
        f = self.functions['f']
        a8 = self.coefficients['a8']
        a9 = self.coefficients['a9']

        nb_ops = rng.randint(5, self.max_ops + 1)

        try:
            # generate an expression and rewrite it,
            # avoid issues in 0 and convert to SymPy
            # here, y and z have the role of constants
            expr = self._generate_expr(nb_ops, self.max_int, rng, require_x=True, require_y=True, require_z=True)
            infix = self.prefix_to_infix(expr)
            expr = self.infix_to_sympy(infix).subs(y, a8).subs(z, a9)

            # skip constant expressions
            if x not in expr.free_symbols or a8 not in expr.free_symbols or a9 not in expr.free_symbols:
                return None

            # reduce expression coefficients
            # simplify constants with coefficients
            # re-index coefficients
            # clean constant terms
            expr = self.reduce_coefficients(expr)
            expr = self.simplify_const_with_coeff(expr, coeffs=[a8, a9])
            expr = self.reindex_coefficients(expr)
            expr = clean_degree2_solution(expr, x, a8, a9)

            # skip constant expressions
            if x not in expr.free_symbols or a8 not in expr.free_symbols or a9 not in expr.free_symbols:
                return None

            # skip invalid expressions
            if has_inf_nan(expr) or has_I(expr):
                return None

            # convert the expression to prefix
            expr_prefix = self.sympy_to_prefix(expr)

            # skip too long expressions
            if len(expr_prefix) + 2 > self.max_len:
                return None

            # skip when the number of operators in f is too far from expected
            real_nb_ops = sum(1 if op in self.OPERATORS else 0 for op in expr_prefix)
            if real_nb_ops < nb_ops / 2:
                return None

            # express the constant a8 in terms of x, f, and a9
            solve_a8 = sp.solve(f(x) - expr, a8, check=False, simplify=False)
            if len(solve_a8) == 0 or type(solve_a8) is not list:
                return None
            solve_a8 = [s for s in solve_a8 if x in s.free_symbols]
            if len(solve_a8) == 0:
                return None
            _a8 = solve_a8[rng.randint(len(solve_a8))]
            if type(_a8) is tuple or type(_a8) is sp.Piecewise:
                return None

            # express the constant a9 in terms of x, f, and f'
            solve_a9 = sp.solve(_a8.diff(x), a9, check=False, simplify=False)
            if len(solve_a9) == 0 or type(solve_a9) is not list:
                return None
            solve_a9 = [s for s in solve_a9 if x in s.free_symbols]
            if len(solve_a9) == 0:
                return None
            _a9 = solve_a9[rng.randint(len(solve_a9))]
            if type(_a9) is tuple or type(_a9) is sp.Piecewise:
                return None

            # compute differential equation
            # simplify the differential equation by removing positive terms
            eq = _a9.diff(x)
            if not eq.has(f(x).diff(x, x)):
                return None
            eq = simplify_equa_diff(eq, required=f(x).diff(x, 2))
            if rng.randint(2) == 1:
                eq = simplify(eq, seconds=1)

            # skip invalid expressions
            if has_inf_nan(eq) or has_I(eq):
                return None

            # convert equation to prefix
            eq_prefix = self.sympy_to_prefix(eq)
            eq_prefix = self.clean_prefix(eq_prefix)

            # skip too long equations
            if len(eq_prefix) + 2 > self.max_len:
                return None

            # perform checks
            check_eq = simplify(eq.subs(f(x), expr).doit(), seconds=1)
            if check_eq != 0:
                eval_values = eval_test_zero(check_eq)
                if len([y for y in eval_values if y <= ZERO_THRESHOLD]) / len(eval_values) < 0.2:
                    return None

        except TimeoutError:
            raise
        except (ValueError, NotImplementedError, AttributeError, RecursionError, ValueErrorExpression, UnknownSymPyOperator):
            return None
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, infix, e.args))
            return None

        return eq_prefix, expr_prefix

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument("--operators", type=str, default="add:2,sub:1",
                            help="Operators (add, sub, mul, div), followed by weight")
        parser.add_argument("--max_ops", type=int, default=10,
                            help="Maximum number of operators")
        parser.add_argument("--max_ops_G", type=int, default=4,
                            help="Maximum number of operators for G in IPP")
        parser.add_argument("--max_int", type=int, default=10000,
                            help="Maximum integer value")
        parser.add_argument("--int_base", type=int, default=10,
                            help="Integer representation base")
        parser.add_argument("--balanced", type=bool_flag, default=False,
                            help="Balanced representation (base > 0)")
        parser.add_argument("--precision", type=int, default=10,
                            help="Float numbers precision")
        parser.add_argument("--positive", type=bool_flag, default=False,
                            help="Do not sample negative numbers")
        parser.add_argument("--rewrite_functions", type=str, default="",
                            help="Rewrite expressions with SymPy")
        parser.add_argument("--leaf_probs", type=str, default="0.75,0,0.25,0",
                            help="Leaf probabilities of being a variable, a coefficient, an integer, or a constant.")
        parser.add_argument("--n_variables", type=int, default=1,
                            help="Number of variables in expressions (between 1 and 4)")
        parser.add_argument("--n_coefficients", type=int, default=0,
                            help="Number of coefficients in expressions (between 0 and 10)")
        parser.add_argument("--clean_prefix_expr", type=bool_flag, default=True,
                            help="Clean prefix expressions (f x -> Y, derivative f x x -> Y')")

    def create_train_iterator(self, task, params, data_path):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            rng=None,
            params=params,
            path=(None if data_path is None else data_path[task][0])
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

    def create_test_iterator(self, data_type, task, params, data_path):
        """
        Create a dataset for this environment.
        """
        assert data_type in ['valid', 'test']
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            rng=np.random.RandomState(0),
            params=params,
            path=(None if data_path is None else data_path[task][1 if data_type == 'valid' else 2])
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=params.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )


class EnvDataset(Dataset):

    def __init__(self, env, task, train, rng, params, path):
        super(EnvDataset).__init__()
        self.env = env
        self.rng = rng
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        assert (train is True) == (rng is None)
        assert task in CharSPEnvironment.TRAINING_TASKS

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        self.same_nb_ops_per_batch = params.same_nb_ops_per_batch

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path)
            logger.info(f"Loading data from {path} ...")
            with io.open(path, mode='r', encoding='utf-8') as f:
                # either reload the entire file, or the first N lines (for the training set)
                if not train:
                    lines = [line.rstrip().split('|') for line in f]
                else:
                    lines = []
                    for i, line in enumerate(f):
                        if i == params.reload_size:
                            break
                        if i % params.n_gpu_per_node == params.local_rank:
                            lines.append(line.rstrip().split('|'))
            self.data = [xy.split('\t') for _, xy in lines]
            self.data = [xy for xy in self.data if len(xy) == 2]
            logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test (default of 5000 if no file provided)
        if self.train:
            self.size = 1 << 60
        else:
            self.size = 5000 if path is None else len(self.data)

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        nb_ops = [sum(int(word in self.env.OPERATORS) for word in seq) for seq in x]
        # for i in range(len(x)):
        #     print(self.env.prefix_to_infix(self.env.unclean_prefix(x[i])))
        #     print(self.env.prefix_to_infix(self.env.unclean_prefix(y[i])))
        #     print("")
        x = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_ops)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.rng is None:
            assert self.train is True
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.rng = np.random.RandomState([worker_id, self.global_rank, self.env_base_seed])
            logger.info(f"Initialized random generator for worker {worker_id}, with seed {[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed}).")

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        if self.train:
            index = self.rng.randint(len(self.data))
        x, y = self.data[index]
        x = x.split()
        y = y.split()
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:

            try:
                if self.task == 'prim_fwd':
                    xy = self.env.gen_prim_fwd(self.rng)
                elif self.task == 'prim_bwd':
                    xy = self.env.gen_prim_bwd(self.rng, predict_primitive=True)
                elif self.task == 'prim_ibp':
                    xy = self.env.gen_prim_ibp(self.rng)
                elif self.task == 'derivative':
                    xy = self.env.gen_prim_bwd(self.rng, predict_primitive=False)
                elif self.task == 'ode1':
                    xy = self.env.gen_ode1(self.rng)
                elif self.task == 'ode2':
                    xy = self.env.gen_ode2(self.rng)
                else:
                    raise Exception(f'Unknown data type: {self.task}')
                if xy is None:
                    continue
                x, y = xy
                break
            except TimeoutError:
                continue
            except Exception as e:
                logger.error("An unknown exception of type {0} occurred for worker {4} in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, 'F', e.args, self.get_worker_id()))
                continue
        self.count += 1

        # clear SymPy cache periodically
        if CLEAR_SYMPY_CACHE_FREQ > 0 and self.count % CLEAR_SYMPY_CACHE_FREQ == 0:
            logger.warning(f"Clearing SymPy cache (worker {self.get_worker_id()})")
            clear_cache()

        return x, y
