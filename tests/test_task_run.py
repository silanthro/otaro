import logging
from pathlib import Path

from otaro.task import Task

logging.basicConfig()
logger = logging.getLogger("otaro.tests.test_task_run")
logger.setLevel(logging.INFO)


def test_task_run():
    task = Task.from_config("tests/math.yml")
    response = task.run(equation="40 + 2")
    assert response.answer == 42


def test_task_optim():
    task = Task.from_config("tests/poet.yml")
    response = task.run(topic="life")
    assert "frog" in response.haiku.lower()
    optimized_file = Path("./tests/poet.optim.yml")
    assert optimized_file.exists() is True
    optimized_desc = task.desc
    # Check that optimization loads by default
    new_task = Task.from_config("tests/poet.yml")
    assert new_task.desc == optimized_desc
    optimized_file.unlink(missing_ok=True)
