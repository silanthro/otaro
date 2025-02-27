# Configure a task in YAML

A `Task` is the basic building block in Otaro. Similar to DSPy, Otaro treats LLM calls like functions with inputs and outputs.

## Quickstart

```yaml title="quickstart.yml"
# Define a task that takes a topic and returns a list of quotes

model: gemini/gemini-2.0-flash-001

inputs:
  topic:
    type: str

outputs:
  quotes:
    type: list
    list_child_type:
      quote:
        type: str
```

```py title="quickstart.py"
from otaro import Field, Task

# Define a task that takes a topic and returns a list of quotes
task = Task.from_config("quickstart.yml")

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

## Adding rules

Instead of trying to figure out the best prompt, define the desired output with `rules`. Then, whenever `task.run` is called, the task will be automatically optimized to enforce the rules where possible.

For illustration, rewrite the task above to enforce 3 quotes:

```yaml title="quickstart.yml" hl_lines="16-18"
# Define a task that takes a topic and returns a list of quotes

model: gemini/gemini-2.0-flash-001

inputs:
  topic:
    type: str

outputs:
  quotes:
    type: list
    list_child_type:
      quote:
        type: str

# Add rule to enforce len(quotes) == 3
rules:
  - otaro.rules.length_eq(quotes, 3)
```

Running the script will automatically optimize the task to output three quotes.

<!-- termynal -->

```console
$ uv run quickstart.py

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

## Adding custom rules

Custom rules can be defined as functions in a Python file and then referenced within the YAML config.

For example, suppose we want all the quotes to be in uppercase.

```py title="custom_rules.py"
def quotes_are_uppercase(sample):
    return all(q.isupper() for q in sample.quotes)
```

```yaml title="quickstart.yml" hl_lines="16-18"
# Define a task that takes a topic and returns a list of quotes

model: gemini/gemini-2.0-flash-001

inputs:
  topic:
    type: str

outputs:
  quotes:
    type: list
    list_child_type:
      quote:
        type: str

# Add custom uppercase rule
rules:
  - custom_rules.quotes_are_uppercase
```

Then we simply run the same script:


<!-- termynal -->

```console
$ uv run quickstart.py

INFO:otaro.task:Evaluating rules
---> 100%
INFO:otaro.task:Rule 1: custom_rules.quotes_are_uppercase - False
INFO:otaro.task:Optimizing task...
INFO:otaro.task:Evaluating prompt: "Write quotes about {topic}. Output all quotes in uppercase."
INFO:otaro.task:Evaluating rules
---> 100%
INFO:otaro.task:Rule 1: custom_rules.quotes_are_uppercase - True
INFO:otaro.task:    Score: 1.0
INFO:otaro.task:All scores: [0.0, 1.0]
INFO:otaro.task:Selecting prompt #1 with score 1.0

THE PEN IS MIGHTIER THAN THE SWORD IF THE SWORD IS VERY SHORT, AND THE PEN IS VERY SHARP.
IN ANCIENT TIMES CATS WERE WORSHIPPED AS GODS; THEY HAVE NOT FORGOTTEN THIS.
WISDOM COMES FROM EXPERIENCE. EXPERIENCE IS OFTEN A RESULT OF LACK OF WISDOM.
```
