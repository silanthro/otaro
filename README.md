# otaro

## Example

Create a config YAML file to describe the task.

```yaml
# A sample config for a haiku writing task

model: gemini/gemini-2.0-flash-001

inputs:
- topic

outputs:
- haiku
```

```py
from otaro import Task

task = Task.from_config("poet.yml")
response = task.run(topic="lion")
print(response.haiku)

"""
Green skin on blue pond,
A croaking song fills the air,
Summer's gentle kiss.
"""
```

See `./examples` for more examples and configs.

## Deploy API

Run `otaro` to deploy a config file as an API.

```
$ otaro examples/poet.yml
```

Navigate to "https://localhost:8000" to view and test the API documentation.
