import json
import logging
from enum import Enum
from types import GenericAlias
from typing import Any, Type, Union, get_args, get_origin

from pydantic import BaseModel

from otaro.parsing import llm_parse_json

logging.basicConfig()
logger = logging.getLogger("otaro.types")
logger.setLevel(logging.INFO)


class FieldParsingError(Exception):
    def __init__(self, message: str, field: Union["Field", None] = None):
        super().__init__(message)
        self.field = field


def get_schema(value_type: Type | GenericAlias):
    if value_type in [str, int, float, bool]:
        return value_type.__name__
    elif issubclass(value_type, Enum):
        return f'{{"enum": {json.dumps(list(map(lambda c: c.name, value_type)))}}}'
    elif value_type == list:
        return '{"items": {}, "type": "array"}'
    elif get_origin(value_type) == list:
        child_type = get_args(value_type)[0]
        if child_type in [str, int, float, bool] or issubclass(child_type, Enum):
            return f'{{"items": {get_schema(child_type)}, "type": "array"}}'
        elif child_type == list or get_origin(child_type) == list:
            child_schema = get_schema(child_type)
            nesting = 0
            while True:
                if f"ListChild_{nesting}" in child_schema:
                    nesting += 1
                else:
                    break
            return f'{{"$defs": {{"ListChild_{nesting}": {child_schema}}}, "items": {{"$ref": "#/$defs/ListChild_{nesting}"}}, "type": "array"}}'
        else:
            return f'{{"$defs": {{"{child_type.__name__.capitalize()}": {get_schema(child_type)}}}, "items": {{"$ref": "#/$defs/{child_type.__name__.capitalize()}"}}, "type": "array"}}'
    elif issubclass(value_type, BaseModel):
        return value_type.model_json_schema()


def get_dummy_value(value_type: Type | GenericAlias):
    if value_type == str:
        return "Foo"
    elif value_type == int:
        return 42
    elif value_type == float:
        return 3.14
    elif value_type == bool:
        return True
    elif issubclass(value_type, Enum):
        return list(map(lambda c: c.name, value_type))[0]
    elif value_type == list:
        # Dummy list with 2 items - TODO: consider using random length
        return ["Foo", "Foo"]
    elif get_origin(value_type) == list:
        # Dummy list with 2 items - TODO: consider using random length
        child_type = get_args(value_type)[0]
        return [get_dummy_value(child_type), get_dummy_value(child_type)]
    elif issubclass(value_type, BaseModel):
        attribute_values = {}
        for attr_name, field_info in value_type.model_fields.items():
            attribute_values[attr_name] = get_dummy_value(field_info.annotation)
        return attribute_values


def parse_string(value_type: Type | GenericAlias, value: str, to_dict: bool = False):
    if value_type in [str, int, float]:
        return value_type(value)
    elif value_type == bool:
        if isinstance(value, str) and value.lower() in [
            "false",
            "f",
            "no",
            "0",
            "null",
            "none",
            "undefined",
        ]:
            return False
        return bool(value)
    elif issubclass(value_type, Enum):
        parsed_value = value_type(value)
        if to_dict:
            return parsed_value.value
        else:
            return parsed_value
    elif value_type == list or get_origin(value_type) == list:
        try:
            value_json = llm_parse_json(value)
        except ValueError as e:
            raise FieldParsingError("Invalid JSON", None) from e
        if not isinstance(value_json, list):
            raise FieldParsingError(
                f"{json.dumps(value_json)} should be a list but is {type(value_json)}"
            )
        parsed_value = []
        if value_type == list:
            child_type = str
        else:
            child_type = get_args(value_type)[0]
        for i, child in enumerate(value_json):
            try:
                parsed_value.append(
                    parse_string(child_type, json.dumps(child), to_dict=to_dict)
                )
            except Exception as e:
                raise FieldParsingError(
                    f"Error parsing child at index {i} - {e}"
                ) from e
        return parsed_value
    elif issubclass(value_type, BaseModel):
        try:
            value_json = llm_parse_json(value)
        except ValueError as e:
            raise FieldParsingError("Invalid JSON", None) from e
        try:
            parsed_value = value_type(**value_json)
        except Exception as e:
            raise FieldParsingError(str(e)) from e
        if to_dict:
            return parsed_value.model_dump(mode="json")
        else:
            return parsed_value


class Field(BaseModel):
    name: str
    type: Any = str
    desc: str = ""
    default: Any = None

    def __init__(
        self,
        name: str,
        type: Type | GenericAlias = str,
        desc: str = "",
        default: Any = None,
    ):
        super().__init__(
            name=name,
            type=type,
            desc=desc,
            default=default,
        )
        # Cast arbitrary list to list[str]
        if self.type == list:
            self.type = list[str]

    @property
    def model(self):
        return self.type
        if self.type in [str, int, float, bool]:
            return self.type
        elif issubclass(self.type, Enum):
            return self.type
        elif self.type == list:
            return self.type
        elif get_origin(self.type) == list:
            return self.type
        elif issubclass(self.type, BaseModel):
            return self.type

    @property
    def schema(self):
        return get_schema(self.type)

    @property
    def simple_schema(self):
        if self.type in [str, int, float, bool]:
            return self.type.__name__
        elif issubclass(self.type, Enum):
            return "enum"
        elif self.type == list:
            return "list"
        elif get_origin(self.type) == list:
            return "list"
        elif issubclass(self.type, BaseModel):
            return "object"

    @property
    def dummy_value(self):
        return get_dummy_value(self.type)

    def __str__(self):
        docstring = f"`{self.name}` ({self.simple_schema})"
        details = [
            self.desc + ("." if len(self.desc) and not self.desc.endswith(".") else "")
        ]
        if issubclass(self.type, Enum):
            details.append("Respond with one of the following enum values.")
            details.append(f"Enum Schema: {self.schema}")
        if (
            self.type == list
            or get_origin(self.type) == list
            or issubclass(self.type, BaseModel)
        ):
            if self.type == list or get_origin(self.type) == list:
                details.append("Respond with a single JSON array.")
            else:
                details.append("Respond with a single JSON object.")
            details.append(f"JSON Schema: {self.schema}")
        details_str = " ".join(details)
        if len(details_str):
            docstring += f": {details_str}"
        return docstring

    @property
    def dummy_template(self):
        return f"[[ ## {self.name} ## ]]\n{{{self.name}}}"

    def template(self, value):
        if isinstance(value, (str, float, int)):
            return f"[[ ## {self.name} ## ]]\n{value}"
        else:
            return f"[[ ## {self.name} ## ]]\n{json.dumps(value)}"

    def parse(self, value: str, to_dict=False):
        try:
            return parse_string(self.type, value, to_dict)
        except Exception as e:
            raise FieldParsingError(
                f"Error parsing {self.name} with schema {self.schema} - {e}",
                self,
            ) from e

    def __call__(self, value):
        return self.parse(value)
