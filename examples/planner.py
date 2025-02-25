import os
from pathlib import Path

from otaro.task import Task


def main():
    file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    task = Task.from_config(file_dir / "planner.yaml")
    response = task.run(
        task="Search Google for news on AI and save it to a CSV", optimize=False
    )
    print("Plan:")
    for step in response.plan:
        print(f"{step.number} [{step.status.value}]: {step.desc}")


if __name__ == "__main__":
    main()
