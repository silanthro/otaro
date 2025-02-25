from pydantic import BaseModel


# For testing
def always_true(_: BaseModel):
    return True


# For testing
def always_false(_: BaseModel):
    return False


def eq(a, b):
    return a == b


def neq(a, b):
    return a != b


def contains(a, b):
    if isinstance(a, str):
        a = a.lower()
    if isinstance(b, str):
        b = b.lower()
    return b in a


def length_eq(a, b: int):
    return len(a) == b
