# Configure a task in Python

A `Task` is the basic building block in Otaro. Similar to DSPy, Otaro treats LLM calls like functions with inputs and outputs.

## Quickstart

```py title="quickstart.py"
from otaro import Field, Task

# Define a task that takes a topic and returns a list of quotes
task = Task(
    model="gemini/gemini-2.0-flash-001",
    inputs=["topic"],
    outputs=[
        Field(
            name="quotes",
            type="list",
            list_child_type="str",
        )
    ],
)

response = task.run(topic="discworld")
for quote in response.quotes:
    print(quote)
```

<!-- termynal -->

```console
$ uv run quickstart.py

The pen is mightier than the sword if the sword is very short, and the pen is very sharp.
In ancient times cats were worshipped as gods; they have not forgotten this.
Wisdom comes from experience. Experience is often a result of lack of wisdom.
Words in the heart cannot be taken.
```

## Defining a task

At minimum, a `Task` needs to be initialized with:

- `#!python model (str)`: A model name, used by `litellm` under the hood
- `#!python inputs (list[str | Field])`: A list of inputs that the task is expected to receive
- `#!python outputs (list[str | Field])`: A list of outputs to be produced

### Fields

A `Field` is used to specify the type of output that the LLM is supposed to generate.

```
TODO: Accept base classes and BaseModel types for type argument
```

The `Field` definitions will also be used to parse the LLM response and return an output with attributes of the correct types e.g. `response.quotes` is of type `list[str]` above.

For convenience, a `Field` can also be defined with a single string, which will be interpreted as the field name and have default attributes e.g. `type="str"`. For example, `inputs` is set as `["topic"]` in `quickstart.py` above.

## Adding rules

Instead of trying to figure out the best prompt, define the desired output with `rules`. Then, whenever `task.run` is called, the task will be automatically optimized to enforce the rules where possible.

For illustration, rewrite the task above to enforce 3 quotes:

```py title="three_quotes.py" hl_lines="14-17"
from otaro import Field, Task

# Create a task that takes a topic and returns a list of quotes
task = Task(
    model="gemini/gemini-2.0-flash-001",
    inputs=["topic"],
    outputs=[
        Field(
            name="quotes",
            type="list",
            list_child_type="str",
        )
    ],
    # Add rule to enforce len(quotes) == 3
    rules=[
        "otaro.rules.length_eq(quotes, 3)",
    ]
)

response = task.run(topic="discworld")
for quote in response.quotes:
    print(quote)
```

Running the script will automatically optimize the task to output three quotes.

<!-- termynal -->

```console
$ uv run three_quotes.py

INFO:otaro.task:Evaluating rules
---> 100%
INFO:otaro.task:Rule 1: otaro.rules.length_eq(quotes, 3) - False
INFO:otaro.task:Optimizing task...
INFO:otaro.task:Evaluating prompt: "Write three quotes about {topic}."
INFO:otaro.task:Evaluating rules
---> 100%
INFO:otaro.task:Rule 1: otaro.rules.length_eq(quotes, 3) - True
INFO:otaro.task:    Score: 1.0
INFO:otaro.task:All scores: [0.0, 1.0]
INFO:otaro.task:Selecting prompt #1 with score 1.0

The pen is mightier than the sword if the sword is very short, and the pen is very sharp.
In ancient times cats were worshipped as gods; they have not forgotten this.
Wisdom comes from experience. Experience is often a result of lack of wisdom.
```