import json
import logging
from enum import Enum
from typing import Union

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from otaro.parsing import llm_parse_json

logging.basicConfig()
logger = logging.getLogger("otaro.types")
logger.setLevel(logging.INFO)


class FieldType(str, Enum):
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    LIST = "list"
    OBJECT = "object"


class Field(BaseModel):
    name: str
    type: FieldType
    desc: str = ""
    enum_members: Union[list[str], None] = None
    list_child_type: Union["Field", None] = None
    object_attributes: Union[list["Field"], None] = None

    def __init__(
        self,
        name: str,
        type: FieldType = FieldType.STR,
        desc: str = "",
        enum_members: Union[list[str], None] = None,
        list_child_type: Union["Field", None] = None,
        object_attributes: list["Field"] | None = None,
    ):
        super().__init__(
            name=name,
            type=type,
            desc=desc,
            enum_members=enum_members,
            list_child_type=list_child_type,
            object_attributes=object_attributes or [],
        )
        self.type = type
        self.enum_members = enum_members or []
        self.enum_members = [m.strip() for m in self.enum_members]
        if list_child_type:
            if isinstance(list_child_type, Field):
                self.list_child_type = list_child_type
            else:
                self.list_child_type = Field(**list_child_type)
        if object_attributes:
            self.object_attributes = [Field(**attr) for attr in object_attributes]

    @property
    def model(self):
        if self.type == FieldType.STR:
            return str
        elif self.type == FieldType.INT:
            return int
        elif self.type == FieldType.FLOAT:
            return float
        elif self.type == FieldType.BOOL:
            return bool
        elif self.type == FieldType.ENUM:
            return Enum(
                self.name,
                [(str(v), v) for v in self.enum_members],
                type=type(
                    self.enum_members[0]
                ),  # Mix type of first member (assuming all members are same type)
            )
        elif self.type == FieldType.LIST:
            return list[self.list_child_type.model]
        elif self.type == FieldType.OBJECT:
            attributes = {}
            for attr in self.object_attributes:
                attributes[attr.name] = (attr.model, FieldInfo())
            model = create_model(self.name, **attributes)
            return model

    @property
    def schema(self):
        if self.type in [
            FieldType.STR,
            FieldType.INT,
            FieldType.FLOAT,
            FieldType.BOOL,
        ]:
            return self.type
        elif self.type == FieldType.ENUM:
            return f'{{"enum": {json.dumps(self.enum_members)}}}'
        elif self.type == FieldType.LIST:
            # TODO: This should be recursive
            if self.list_child_type.type in [
                FieldType.STR,
                FieldType.INT,
                FieldType.FLOAT,
                FieldType.BOOL,
            ]:
                return f'{{"items": {self.list_child_type.type}, "type": "array"}}'
            if self.list_child_type.type == FieldType.ENUM:
                return f'{{"items": {self.list_child_type.schema}, "type": "array"}}'
            else:
                return f'{{"$defs": {{"{self.list_child_type.name.capitalize()}": {self.list_child_type.schema}}}, "items": {{"$ref": "#/$defs/{self.list_child_type.name.capitalize()}"}}, "type": "array"}}'
        elif self.type == FieldType.OBJECT:
            return self.model.model_json_schema()

    @property
    def dummy_value(self):
        if self.type == FieldType.STR:
            return "Foo"
        elif self.type == FieldType.INT:
            return 42
        elif self.type == FieldType.FLOAT:
            return 3.14
        elif self.type == FieldType.BOOL:
            return True
        elif self.type == FieldType.ENUM:
            return self.enum_members[0]
        elif self.type == FieldType.LIST:
            return [self.list_child_type.dummy_value]
        elif self.type == FieldType.OBJECT:
            attribute_values = {}
            for attr in self.object_attributes:
                attribute_values[attr.name] = attr.dummy_value
            return attribute_values

    def __str__(self):
        docstring = f"`{self.name}` ({self.type})"
        details = [
            self.desc + ("." if len(self.desc) and not self.desc.endswith(".") else "")
        ]
        if self.type == FieldType.ENUM:
            details.append("Respond with one of the following enum values.")
            details.append(f"Enum Schema: {self.schema}")
        if self.type in [FieldType.LIST, FieldType.OBJECT]:
            if self.type == FieldType.LIST:
                details.append("Respond with a single JSON array.")
            elif self.type == FieldType.OBJECT:
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
        if self.type == FieldType.STR:
            return value
        elif self.type == FieldType.INT:
            return int(value)
        elif self.type == FieldType.FLOAT:
            return float(value)
        elif self.type == FieldType.BOOL:
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
        elif self.type == FieldType.ENUM:
            parsed_value = self.model(value)
            if to_dict:
                return parsed_value.value
            else:
                return parsed_value
        elif self.type == FieldType.LIST:
            value_json = llm_parse_json(value)
            return [
                self.list_child_type.parse(json.dumps(c), to_dict=to_dict)
                for c in value_json
            ]
        elif self.type == FieldType.OBJECT:
            value_json = llm_parse_json(value)
            parsed_value = self.model(**value_json)
            if to_dict:
                return parsed_value.model_dump(mode="json")
            else:
                return parsed_value

    def __call__(self, value):
        return self.parse(value)


# OPERATORS = {
#     "<": lambda a, b: a < b,
#     ">": lambda a, b: a > b,
#     "<=": lambda a, b: a <= b,
#     ">=": lambda a, b: a >= b,
#     "!=": lambda a, b: a != b,
#     "==": lambda a, b: a == b,
#     "in": lambda a, b: a in b,
#     "not": lambda a, _: not a,
#     "and": lambda a, b: a and b,
#     "or": lambda a, b: a or b,
#     "len": lambda a, _: len(a),
# }


# class RuleOperand(BaseModel):
#     value: str
#     type: str

#     def __init__(
#         self,
#         type: str = "",
#         value: str = "",
#         **kwargs,
#     ):
#         super().__init__(
#             type=type,
#             value=value,
#         )
#         self.type = type

#     def __str__(self):
#         return f"{self.type}<{self.value}>"

#     def eval(self, result=None):
#         if self.type == "field":
#             props = self.value.split(".")
#             subresult = result.model_dump(mode="json")
#             for prop in props:
#                 match = re.match(re.compile("(?P<prop>.*?)\\[(?P<idx>\\d*)\\]"), prop)
#                 if match:
#                     prop = match.groupdict()["prop"]
#                     idx = int(match.groupdict()["idx"])
#                     subresult = subresult[prop][idx]
#                 else:
#                     subresult = subresult[prop]
#             return subresult
#         if self.type == "str":
#             return self.value
#         elif self.type == "float":
#             return float(self.value)
#         elif self.type == "int":
#             return int(self.value)


# class Rule(BaseModel):
#     a: Union[RuleOperand, "Rule"]
#     operator: str
#     b: Union[RuleOperand, "Rule", None] = None
#     type: str

#     def __init__(
#         self,
#         a: Union[RuleOperand, "Rule", None] = None,
#         operator: str = "",
#         b: Union[RuleOperand, "Rule", None] = None,
#         type: str = "",
#         **kwargs,
#     ):
#         super().__init__(
#             a=a,
#             operator=operator,
#             b=b,
#             type=type,
#         )
#         if self.a.type == "rule":
#             self.a = Rule(**a)
#         if self.b and self.b.type == "rule":
#             self.b = Rule(**b)

#     def eval(self, result=None):
#         parsed_a = self.a.eval(result)
#         if self.b:
#             parsed_b = self.b.eval(result)
#         else:
#             parsed_b = None
#         return OPERATORS[self.operator](parsed_a, parsed_b)

#     def __str__(self):
#         if self.operator in ["not", "len"]:
#             return f"{self.operator}({self.a})"
#         else:
#             return f"({self.a}) {self.operator} ({self.b})"


# class TaskParams(BaseModel):
#     desc: str = ""
#     inputs: list[Field] = []
#     outputs: list[Field] = []
#     rules: list[str] = []
#     data: list[dict] = []


# class OptimParams(BaseModel):
#     desc: str = ""
#     inputs: list[Field] = []
#     outputs: list[Field] = []
#     rules: list[str] = []
#     data: list[dict] = []
