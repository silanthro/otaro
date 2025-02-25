import logging

from otaro.task import Task

logging.basicConfig()
logger = logging.getLogger("otaro.tests.test_task_load")
logger.setLevel(logging.INFO)


with open("./tests/math_system_prompt.txt", "r") as file:
    MATH_SYSTEM_PROMPT = file.read()

with open("./tests/nested_system_prompt.txt", "r") as file:
    NESTED_SYSTEM_PROMPT = file.read()

with open("./tests/nested_user_prompt.txt", "r") as file:
    NESTED_USER_PROMPT = file.read()


def test_load_config():
    task = Task.from_config("tests/math.yaml")
    assert task.desc == ""
    assert len(task.inputs) == 1
    assert len(task.outputs) == 1
    prompt_template = task.prompt_template
    assert prompt_template["messages"][0]["content"] == MATH_SYSTEM_PROMPT


def test_load_nested():
    task = Task.from_config("tests/nested.yaml")
    system_prompt = task.prompt_template["messages"][0]["content"]
    # logger.info(system_prompt)
    dummy_input = task.dummy_input
    # logger.info(dummy_input)
    user_prompt = task.get_prompt(**dummy_input)["messages"][1]["content"]
    # logger.info(user_prompt)
    assert system_prompt == NESTED_SYSTEM_PROMPT
    assert user_prompt == NESTED_USER_PROMPT

    for input_field in task.inputs:
        input_field.parse(str(input_field.dummy_value))
