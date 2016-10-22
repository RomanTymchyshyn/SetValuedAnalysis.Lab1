"""Module that provides functionality for operations with functions."""

import numbers
import operator

class Operable:
    """Decorator that allows to apply standard (as add, multiply etc.) operators to functions."""

    def __init__(self, f):
        """Initialize instance with function or number."""
        self.function = (lambda *args: f) if isinstance(f, numbers.Number) else f

    def __call__(self, *args):
        """Invoke with passed arguments."""
        return self.function(*args)


def _op_to_function_op(oper):
    """Convert operator to operator with functions."""
    def function_op(self, operand):
        """Converted operator."""
        def func(*args):
            """Result function."""
            return oper(self(*args), Operable(operand)(*args))
        return Operable(func)
    return function_op

for name, op in [(name, getattr(operator, name)) for name in dir(operator) if "__" in name]:
    try:
        op(1, 2)
    except TypeError:
        pass
    else:
        setattr(Operable, name, _op_to_function_op(op))
