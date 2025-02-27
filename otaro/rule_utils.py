import importlib
import inspect
import logging
import re

from pydantic import BaseModel
from pyparsing import (
    Literal,
    QuotedString,
    Word,
    delimited_list,
    identbodychars,
    identchars,
    nums,
    one_of,
)

logging.basicConfig()
logger = logging.getLogger("otaro.rule_utils")
logger.setLevel(logging.INFO)

string_arg = QuotedString("'", esc_char="\\", unquote_results=False) | QuotedString(
    '"', esc_char="\\", unquote_results=False
)

number_arg = Word(nums) | Word(nums) + "." + Word(nums)

boolean_arg = Literal("True") | Literal("False")

array_item = string_arg | number_arg
array_list = delimited_list(array_item, allow_trailing_delim=True)
array_arg = Literal("[") + array_list + Literal("]")
tuple_arg = Literal("(") + array_list + Literal(")")

arg_name = Word(identchars, identbodychars)
arg_value = string_arg | number_arg | boolean_arg | tuple_arg | array_arg
generic_arg_value = string_arg | number_arg | boolean_arg | tuple_arg | array_arg
arg_item = (arg_name + Literal("=").suppress() + arg_value) | arg_value
arg_list = delimited_list(arg_item)


def get_value_parser(sample: BaseModel | None = None):
    if sample is None:
        sample_attrs = []
    else:
        sample_attrs = list(
            sample.model_dump(mode="json").keys()
        )  # Note this does not support nested keys

    def value_parser(string, location, tokens):
        key = None
        emit_tokens = []
        if tokens[0] == "[":
            # Arrays
            emit_tokens = eval("[" + ",".join(tokens[1:-1]) + "]")
        elif tokens[0] == "(":
            # Tuples
            emit_tokens = eval("(" + ",".join(tokens[1:-1]) + ")")
        elif tokens[0].startswith('"') or tokens[0].startswith("'"):
            # Quoted strings
            emit_tokens = eval(tokens[0])
        elif tokens[0] in sample_attrs:
            # Sample attributes
            emit_tokens = sample.model_dump(mode="json")[tokens[0]]
        elif len(tokens) > 2 and tokens[1] == "=":
            # Kwarg
            key = tokens[0]
            emit_tokens = value_parser(None, None, tokens[2:])[1]
        else:
            # Everything else
            emit_tokens = eval(tokens[0])

        return [key, emit_tokens]

    return value_parser


# TODO: Supported nested inputs / outputs
def process_signature(args_str: str, sample: BaseModel | None = None):
    if sample is None:
        sample_attrs = []
    else:
        sample_attrs = list(
            sample.model_dump(mode="json").keys()
        )  # Note this does not support nested keys
    field_arg = one_of(sample_attrs)
    arg_value = generic_arg_value | field_arg
    arg_item = (arg_name + Literal("=") + arg_value) | arg_value
    arg_item.set_parse_action(get_value_parser(sample))
    arg_list = delimited_list(arg_item)
    arr = arg_list.parse_string(args_str, parse_all=True)
    args = []
    kwargs = {}
    for i in range(0, len(arr), 2):
        key = arr[i]
        value = arr[i + 1]
        if key is None:
            args.append(value)
        else:
            kwargs[key] = value
    return args, kwargs


# TODO
# def validate_rule_str(rule_str: str, input_fields: list, output_fields: list):
#     raise NotImplementedError


def eval_rule_str(rule_str: str, sample: BaseModel | None = None):
    fn_pattern = re.compile(
        "^(?P<module>.*?)\\.(?P<fn>[a-zA-Z_]*?)(\\((?P<args>.*)\\))?$", re.DOTALL
    )
    match = fn_pattern.match(rule_str.strip())
    if match:
        module_name = match.groupdict().get("module")
        fn_name = match.groupdict().get("fn")
        args_str = match.groupdict().get("args")
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
        if args_str:
            args, kwargs = process_signature(args_str, sample)
            return fn(*args, **kwargs)
        return fn(sample)
    raise ValueError(f"Invalid rule - {rule_str}")


def get_rule_source(rule_str: str):
    fn_pattern = re.compile(
        "^(?P<module>.*?)\\.(?P<fn>[a-zA-Z_]*?)(\\((?P<args>.*)\\))?$", re.DOTALL
    )
    match = fn_pattern.match(rule_str.strip())
    if match:
        module_name = match.groupdict().get("module")
        fn_name = match.groupdict().get("fn")
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
        return f"Rule: {rule_str}\n{inspect.getsource(fn)}"

    raise ValueError(f"Invalid rule - {rule_str}")
