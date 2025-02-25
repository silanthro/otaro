import logging

logging.basicConfig()
logger = logging.getLogger("otaro.task_utils")
logger.setLevel(logging.INFO)


# TODO: Provide more detailed error messages
def parse_attr_config(attr_json: dict):
    if attr_json.get("type") == "enum":
        enum_members = attr_json.get("enum_members", [])
        if not len(enum_members):
            raise ValueError("No enum_members provided")
    if attr_json.get("type") == "list":
        list_child_type_config = attr_json.get("list_child_type")
        if list_child_type_config:
            keys = list(list_child_type_config.keys())
            if len(keys) != 1:
                raise ValueError("Invalid list_child_type")
            attr_json["list_child_type"] = {
                "name": keys[0],
                **parse_attr_config(list_child_type_config[keys[0]]),
            }
        else:
            raise ValueError("No list_child_type provided")
    if attr_json.get("type") == "object":
        object_attributes_config = attr_json.get("object_attributes")
        if object_attributes_config:
            attr_json["object_attributes"] = [
                {"name": name, **parse_attr_config(attrs)}
                for name, attrs in object_attributes_config.items()
            ]
        else:
            raise ValueError("No object_attributes provided")

    return attr_json


def parse_fields_config(json: dict):
    return [
        {
            "name": name,
            **parse_attr_config(attr if attr else {}),
        }
        for name, attr in json.items()
    ]
