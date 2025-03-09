import logging
import re
from enum import Enum

from pydantic import create_model
from pydantic.fields import FieldInfo

from otaro.parsing import llm_parse_json

logging.basicConfig()
logger = logging.getLogger("otaro.task_utils")
logger.setLevel(logging.INFO)


def parse_type_str(
    field_name: str,
    type_str: str | list[str] | dict,
    custom_types: dict | None = None,
):
    basic_types = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }
    if type_str in list(basic_types.keys()):
        return basic_types[type_str]
    elif isinstance(type_str, list):
        return Enum(field_name, [(t, t) for t in type_str])
    elif type_str == "list":
        return list
    elif isinstance(type_str, str) and type_str.startswith("list["):
        rgx_pattern = re.compile("^list\\[(?P<child_type>.*?)\\]$", re.DOTALL)
        match = rgx_pattern.match(type_str)
        if match:
            child_type_str = match.groupdict()["child_type"]
            if "," in child_type_str:
                child_type_str = llm_parse_json(f"[{child_type_str}]")
            return list[parse_type_str("", child_type_str, custom_types)]
        else:
            raise ValueError(f"Invalid list definition - {type_str}")
    elif custom_types and type_str in list(custom_types.keys()):
        return custom_types[type_str]
    elif isinstance(type_str, dict):
        for k, v in type_str.items():
            type_str[k] = (parse_type_str(k, v, custom_types), FieldInfo())
        return create_model(field_name, **type_str)
    else:
        raise ValueError(f"Invalid type definition - {type_str}")


# TODO: Provide more detailed error messages
def parse_attr_config(field_config: dict | str, custom_types: dict | None = None):
    if isinstance(field_config, str):
        return field_config
    elif len(list(field_config.items())) == 1:
        # Parse field_name: type
        field_name, type_str = list(field_config.items())[0]
        return {
            "name": field_name,
            "type": parse_type_str(field_name, type_str, custom_types),
        }
    else:
        # Parse field args
        return {
            "name": field_config["name"],
            "type": parse_type_str(
                field_config["name"], field_config["type"], custom_types
            ),
            "desc": field_config.get("desc", ""),
            "default": field_config.get("default", None),
        }


def parse_fields_config(fields: list, custom_types: dict | None = None):
    return [parse_attr_config(field, custom_types) for field in fields]


# TODO: Support self-referential type
def parse_custom_types(custom_types: list[dict]):
    parsed_types = {}
    for custom_type in custom_types:
        name, attributes = list(custom_type.items())[0]
        for k, v in attributes.items():
            attributes[k] = (parse_type_str("", v, parsed_types), FieldInfo())
        parsed_types[name] = create_model(name, **attributes)
    return parsed_types
