import os
from pathlib import Path

from otaro import Task


def main():
    file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    config_file = file_dir / "error.yml"
    task = Task.from_config(config_file)
    """
    First time the task is run, it should run into a formatting error
    because gpt-4o-mini outputs quotes as a list of strings instead
    of a nested array. The error will be logged but this should be
    fixed automatically.
    """
    response = task.run(topic="discworld", optimize=False)
    for quote in response.quotes:
        print([quote])
    """
    The second time the task is run, the error should not occur again
    since the fix is in place.
    """
    response = task.run(topic="frogs", optimize=False)
    for quote in response.quotes:
        print([quote])
    """
    Remove optimized config
    Comment out lines below to view the optimized config with error
    correction
    """
    # optimized_config_file = file_dir / "error.optim.yml"
    # optimized_config_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
