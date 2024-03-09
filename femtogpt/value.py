# Credits to Andrej Karpathy for the original code (https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py)

import math
import functools
import collections

counter = 0


def next_id():
    global counter
    counter += 1
    return counter

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0

        self._prev = set(_children)
        self._backward = lambda: None
        global counter

        self._op = _op

    def getdata(self):
        if self._op == '':
            return str(self.data)
        return f"data[{self._id}]"

    def set(self, val):
        if self._op == '':
            raise RuntimeError("Can't set constant")
        return f"{self.getdata()} = {val};"
    
    def make_exp(self, val, exp):
        if exp == 0:
            return "1"
        elif exp == 1:
            return val
        elif exp == -1:
            return f"((double)1)/{val}"
        return f"pow({val}, {exp})"
     
    def compile(self):
        if self._op in ('', 'weight', 'bias', 'input'):
            # Set once at init time and thereafter reset in update
            return ""
        if self._op == '*':
            assert len(self._prev) == 2
            return self.set(f"{self._prev[0].getdata()}*{self._prev[1].getdata()}")
        if self._op == '+':
            assert len(self._prev) == 2
            return self.set(f"{self._prev[0].getdata()}+{self._prev[1].getdata()}")
        if self._op == 'ReLU':
            assert len(self._prev) == 1
            return self.set(f"relu({self._prev[0].getdata()})")
        if self._op.startswith('**'):
            exponent = int(self._op[2:])
            assert len(self._prev) == 1
            return self.set(self.make_exp(self._prev[0].getdata(), exponent))
        if self._op == 'exp':
            return self.set(f"exp({self._prev[0].getdata()})")
        if self._op == 'log':
            return self.set(f"log({self._prev[0].getdata()})")
        raise NotImplementedError(self._op)
   
    def getgrad(self):
        if self._op in ('', 'input'):
            raise RuntimeError("Grad for constants and input data not stored")
        return f"grad[{self._id}]"

    def setgrad(self, val):
        if self._op in ('', 'input'):
            return []
        return [f"{self.getgrad()} += clip({val});"]
    
    def backward_compile(out):
        if not out._prev:
            assert out._op in ('', 'weight', 'bias', 'input')
            # Nothing to propagate to children.
            assert not out._prev
            return []
        if out._op == '*':
            self, other = out._prev
            return self.setgrad(f"{other.getdata()}*{out.getgrad()}") +\
                    other.setgrad(f"{self.getdata()}*{out.getgrad()}")
        if out._op == '+':
            self, other = out._prev
            return self.setgrad(f"{out.getgrad()}") + other.setgrad(f"{out.getgrad()}")
        if out._op == 'ReLU':
            self, = out._prev
            return self.setgrad(f"({out.getdata()}>0)*{out.getgrad()}")
        if out._op.startswith('**'):
            exponent = int(out._op[2:])
            self, = out._prev
            exp = out.make_exp(self.getdata(), exponent-1)
            return self.setgrad(f"{exponent}*{exp}*{out.getgrad()}")
        if out._op == 'exp':
            self, = out._prev
            return self.setgrad(f"exp({self.getdata()})*{out.getgrad()}")
        if out._op == 'log':
            self, = out._prev
            return self.setgrad(f"1.0L/{self.getdata()}*{out.getgrad()}")
        raise NotImplementedError(out._op)

    def find(self):
        op = self
        while isinstance(op, Value):
            next = op.forwarded
            if next is None:
                return op
            op = next
        return op

    def args(self):
        return [v.find() for v in self._prev]

    def arg(self, idx):
        return self._prev[idx].find()

    def make_equal_to(self, other):
        self.find().set_forwarded(other)

    def set_forwarded(self, other):
        if self._op == '':
            assert self.data == other.data
        else:
            self.forwarded = other


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def sqrt(self):
        out = Value(math.sqrt(self.data), (self,), "sqrt")

        def _backward():
            self.grad += (0.5 * self.data**-0.5) * out.grad

        out._backward = _backward

        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        # sort the nodes in topological order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for n in reversed(topo):
            n._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    


OPT_LOG = collections.Counter()


def optimize_one(v):
    if v._op == "+":
        args = v.args()
        if any(arg._op == "+" for arg in args):
            OPT_LOG["flatten_plus"] += 1
            new_args = []
            for arg in args:
                if arg._op == "+":
                    new_args.extend(arg.args())
                else:
                    new_args.append(arg)
            v.make_equal_to(Value(0, tuple(new_args), "+"))
            return True
        if len(args) == 1:
            OPT_LOG["plus_single"] += 1
            v.make_equal_to(args[0])
            return True
    return False


@functools.lru_cache(maxsize=None)
def hashcons_array(vs):
    return Array(vs)


def run_optimize_one(v):
    topo = v.topo()
    changed = False
    for op in topo:
        changed |= optimize_one(op.find())
    return changed


class Array(Value):
    def __init__(self, data):
        super().__init__(0, data, "array")
        self._id = next_id()

    def __repr__(self):
        return f"Array({self._prev})"


class Dot(Value):
    def __init__(self, left, right):
        super().__init__(0, (left, right), "dot")
        assert len(left._prev) == len(right._prev)
        self._id = next_id()

        # TODO(max): Figure out a way to compute this automatically using chain
        # rule.
        def _backward():
            left = self._prev[0].find()
            right = self._prev[1].find()
            for i in range(left._prev):
                left._prev[i].grad += right._prev[i].data * self.grad
                right._prev[i].grad += left._prev[i].data * self.grad

        self._backward = _backward

    def __repr__(self):
        return f"Dot(left={self._left}, right={self._right})"


def optimize(v):
    while changed := run_optimize_one(v):
        pass
    topo = v.find().topo()
    for op in topo:
        args = op.args()
        if op._op == "+" and any(arg._op == "*" for arg in args):
            mul_args = tuple(arg for arg in args if arg._op == "*")
            assert all(len(arg._prev) == 2 for arg in mul_args)
            mul_left = hashcons_array(tuple(arg.arg(0) for arg in mul_args))
            mul_right = hashcons_array(tuple(arg.arg(1) for arg in mul_args))
            other_args = tuple(arg for arg in args if arg._op != "*")
            op.make_equal_to(Value(0, (Dot(mul_left, mul_right), *other_args), "+"))
            changed = True
            continue
    return changed


def fmt(v):
    return f"v{v._id}"


def pretty(v):
    topo = v.topo()
    for op in topo:
        if op._op == "input":
            print(f"{fmt(op)} = input")
        elif op._op == "":
            print(f"{fmt(op)} = {op.data}")
        else:
            print(f"{fmt(op)} = {op._op} {' '.join(fmt(c) for c in op.args())}")


def count(v):
    c = collections.Counter()
    for op in v.topo():
        c[op._op] += 1
    return c


def compile(v):
    for op in v.topo():
        if op._op == "dot":
            n = len(op._prev[0]._prev)
            args = op.args()
            print(f"double {fmt(op)} = dot{n}({fmt(args[0])}, {fmt(args[1])});")
        elif op._op == "+":
            print(f"double {fmt(op)} = {' + '.join(fmt(v) for v in op.args())};")
        elif op._op == "array":
            n = len(op._prev)
            print(
                f"double {fmt(op)}[{n}] = {{ {', '.join(fmt(v) for v in op.args())} }};"
            )
        elif op._op == "":
            print(f"double {fmt(op)} = {op.data};")
        elif op._op == "input":
            print(f"double {fmt(op)} = in[{op.data}];")
        elif op._op == "ReLU":
            arg = fmt(op.arg(0))
            print(f"double {fmt(op)} = {arg} > 0 ? {arg} : 0;")
        else:
            raise RuntimeError(f"unexpected op {op._op!r}")
