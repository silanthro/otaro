import os
from pathlib import Path

from otaro.task import Task


def main():
    file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    task = Task.from_config(file_dir / "poet.yaml")
    response = task.run(topic="lion", optimize=False)
    print(f"Haiku:\n{response.haiku}")


if __name__ == "__main__":
    main()
