# Access the prompt

The prompt templates used in Otaro are adapted from DSPy. For any Task, the prompt sent to the LLM can be accessed in two ways

- `Task.prompt_template`: This attribute returns the prompt messages as a template pending input values
- `Task.get_prompt(**input_kwargs)`: This method returns the prompt messages populated with values from input_kwargs

In both cases, the prompt is returned as a list of messages, including the system prompt, user request, and demos if provided.

See the next section for an example.

# Example prompt

For illustration, the following config will be used.

```yaml title="quotes.yml"
# Define a task that takes a topic and returns a list of quotes

model: gemini/gemini-2.0-flash-001

inputs:
- topic

outputs:
- quotes: list[str]

demos:
- topic: frogs
  quotes: [
    "Kissing a frog to get the prince is a waste of a perfectly good frog.",
    "Every journey begins with a single hop."
  ]
```

The following describes what is shown if we run the following code to access the prompt:

```py
from otaro import Task

task = Task.from_config("quotes.yml")

# View the prompt with `topic="life"`
for message in task.get_prompt(topic="life")["messages"]:
    print(message["content"])
```

First, the System Prompt includes the description of input and output fields.

```title="System Prompt i.e. role='system'"
Your input fields are:
1. `topic` (str)

Your output fields are:
1. `reasoning` (str)
2. `quotes` (list):  Respond with a single JSON array. JSON Schema: {"items": str, "type": "array"}

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## topic ## ]]
{topic}

[[ ## reasoning ## ]]
{reasoning}

[[ ## quotes ## ]]
{quotes}

[[ ## completed ## ]]

In adhering to this structure, your objective is:
    Given the field topic, produce the field quotes.
```

Thereafter, if any demos are provided, we append them as user/assistant messages.


```title="Demo 1 (role='user')"
[[ ## topic ## ]]
frogs

Task: 

Start your reasoning by reiterating the task, discuss any common errors, highlight important rules, and then reasoning about the inputs.

Respond with the corresponding output fields reasoning, quotes, and then ending with the marker for `completed`.
```

```title="Demo 1 (role='assistant')"
[[ ## quotes ## ]]
["Kissing a frog to get the prince is a waste of a perfectly good frog.", "Every journey begins with a single hop."]

[[ ## completed ## ]]
```

Finally, we end off with the actual user request.

```title="User Request (role='user')"
[[ ## topic ## ]]
life

Task: 

Start your reasoning by reiterating the task, discuss any common errors, highlight important rules, and then reasoning about the inputs.

Respond with the corresponding output fields reasoning, quotes, and then ending with the marker for `completed`.
```
