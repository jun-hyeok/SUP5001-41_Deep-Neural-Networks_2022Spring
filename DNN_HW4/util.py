import builtins

GRAY = "\033[90m"
RED = "\033[91m"
GREEN = "\033[92m"


def print(*args, **kwargs):
    color = kwargs.pop("color", None)
    input = kwargs.pop("input", False)
    if color:
        builtins.print(color, end="")
        builtins.print(">>>" if input else "", end="")
        builtins.print(*args, **kwargs)
        builtins.print("\033[0m", end="")
    else:
        builtins.print(">>>" if input else "", end="")
        builtins.print(*args, **kwargs)
