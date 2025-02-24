import logging
import re

import yaml
from litellm import acompletion, token_counter
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from tqdm import tqdm

from acai.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from acai.providers import APIModel
from acai.types import Field, Rule

logging.basicConfig()
logger = logging.getLogger("acai.task")
logger.setLevel(logging.INFO)


class CompletionResponse(BaseModel):
    content: str
    num_input_tokens: int
    num_output_tokens: int


def count_tokens(
    message: str,
    role="user",
    model: str | None = None,
):
    return token_counter(
        model=model or APIModel.DEFAULT,
        messages=[
            {
                "role": role,
                "content": message,
            }
        ],
    )


async def _completion(
    messages: list[dict],
    model: str | None = None,
):
    model = model or APIModel.DEFAULT
    response = await acompletion(
        model=model,
        messages=messages,
        num_retries=3,
        timeout=60,
    )
    content = response.choices[0].message.content
    num_input_tokens = token_counter(
        model=model,
        messages=messages,
    )
    num_output_tokens = token_counter(
        model=model,
        messages=[
            {
                "role": "assistant",
                "content": content,
            }
        ],
    )
    return CompletionResponse(
        content=content,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
    )


class Task(BaseModel):
    desc: str = ""
    inputs: list[Field] = []
    outputs: list[Field] = []
    rules: list[Rule] = []

    def __init__(
        self,
        desc: str = "",
        inputs: list[Field] = None,
        outputs: list[Field] = None,
        rules: list[Rule] = None,
        **kwargs,
    ):
        super().__init__(
            desc=desc,
            inputs=inputs or [],
            outputs=outputs or [],
            rules=rules or [],
        )

    @classmethod
    def from_config(cls, config_file: str):
        with open(config_file) as file:
            config = yaml.safe_load(file)
        config["inputs"] = [
            {
                "name": name,
                **(attr if attr else {}),
            }
            for name, attr in config.get("inputs", {}).items()
        ]
        config["outputs"] = [
            {
                "name": name,
                **(attr if attr else {}),
            }
            for name, attr in config.get("outputs", {}).items()
        ]
        return cls(**config)

    @property
    def formatted_desc(self):
        default_desc = f"Given the field{'s' if len(self.inputs) > 1 else ''} {', '.join(f.name for f in self.inputs)}, produce the field{'s' if len(self.outputs) > 1 else ''} {', '.join(f.name for f in self.outputs)}"
        if self.desc:
            return f"{self.desc} {default_desc}"
        else:
            return default_desc

    async def run(self, evaluate=False, **kwargs):
        input_fields = "\n".join(
            [f"{i}. {field}" for i, field in enumerate(self.inputs, 1)]
        )
        output_fields = "\n".join(
            ["1. `reasoning` (str)"]
            + [f"{i}. {field}" for i, field in enumerate(self.outputs, 2)]
        )
        interaction_format = "\n\n".join(
            [field.dummy_template for field in self.inputs]
            + ["[[ ## reasoning ## ]]\n{reasoning}"]
            + [field.dummy_template for field in self.outputs]
        )
        system_prompt = SYSTEM_PROMPT.format(
            input_fields=input_fields,
            output_fields=output_fields,
            interaction_format=interaction_format,
            objective=self.formatted_desc,
        )

        user_prompt = USER_PROMPT.format(
            input_values="\n\n".join(
                field.template(kwargs[field.name]) for field in self.inputs
            ),
            output_fields=", ".join(["reasoning"] + [f.name for f in self.outputs]),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await _completion(messages)

        output_field_names = ["reasoning"] + [field.name for field in self.outputs]
        rgx_patterns = [
            f"\\[\\[ #?#? ?{field_name} #?#? ?\\]\\](?P<{field_name}>.*?)"
            for field_name in output_field_names
        ]
        rgx_pattern = (
            ".*?" + "".join(rgx_patterns) + "\\[\\[ #?#? ?completed #?#? ?\\]\\]"
        )

        rgx = re.compile(rgx_pattern, re.DOTALL)

        match = rgx.match(response.content)
        if match is None:
            print(f"No match on {response.content}")
        else:
            try:
                input_field_attributes = {}
                for field in self.inputs:
                    input_field_attributes[field.name] = (field.model, FieldInfo())
                output_field_attributes = {}
                for field in self.outputs:
                    output_field_attributes[field.name] = (field.model, FieldInfo())
                model = create_model(
                    "Output",
                    reasoning=(str, FieldInfo()),
                    **output_field_attributes,
                    **input_field_attributes,
                    num_input_tokens=(int, FieldInfo()),
                    num_output_tokens=(int, FieldInfo()),
                )
                groupdict = {
                    "num_input_tokens": response.num_input_tokens,
                    "num_output_tokens": response.num_output_tokens,
                }
                for k, v in match.groupdict().items():
                    if k == "reasoning":
                        output_field = str
                    else:
                        output_field = [o for o in self.outputs if o.name == k][0]
                    groupdict[k] = output_field(v.strip())
                result = model(**groupdict, **kwargs)
            except Exception as e:
                result = str(e)
                logger.exception(e)

        evals = [False] * len(self.rules)
        if not isinstance(result, str):
            # Evaluate result against rules
            if evaluate and len(self.rules):
                logger.info("Evaluating rules")
                for i, rule in enumerate(self.rules, 1):
                    outcome = rule.eval(result)
                    logger.info(f"Rule {i}: {rule} - {outcome}")
                    evals[i - 1] = outcome

        if evaluate:
            return {
                "result": result,
                "evals": evals,
            }
        else:
            return result

    async def evaluate(self, data: list):
        total_score = 0
        num_evals = 0
        for sample in tqdm(data):
            try:
                result = await self.run(**sample)
                rule_evals = []
                for rule in self.rules:
                    rule_evals.append(rule.eval(result))
                if all(rule_evals):
                    total_score += 1.0
            except Exception as e:
                logger.info(e)
            num_evals += 1.0
        return total_score / num_evals

    async def optimize(self, data: list, say=None):
        def _log(message: str):
            if say:
                say(message)
            logger.info(message)

        _log("Optimizing task...")

        # Run initial evaluation with rules to determine success rate
        initial_score = await self.evaluate(data)
        _log(f"Initial score: {initial_score}")

        # Get prompts
        prompts = await get_prompts(self, num_prompts=5)

        # For each prompt, run evaluation
        # TODO: Add prompt-demo combination
        scores = []
        for prompt in prompts:
            _log(f"Evaluating prompt: {prompt}")
            task_variant = self.model_copy(deep=True)
            task_variant.desc = prompt
            score = await task_variant.evaluate(data)
            scores.append(score)
            _log(f"\tScore: {score}")
            _log(f"All scores: {[initial_score] + scores}")

        # Select best parameters
        if max(scores) >= initial_score:
            all_scores = [initial_score] + scores
            all_prompts = [self.desc] + prompts

            # Select best-performing prompt that is the shortest
            best_idxs = [
                i for i, score in enumerate(all_scores) if score >= max(all_scores)
            ]
            best_prompts = [all_prompts[i] for i in best_idxs]
            prompt_lengths = [count_tokens(p) for p in best_prompts]
            shortest_idxs = [
                i
                for i, length in enumerate(prompt_lengths)
                if length <= min(prompt_lengths)
            ]
            best_prompt = best_prompts[shortest_idxs[0]]
            best_idx = all_prompts.index(best_prompt)
            _log(f"Selecting prompt #{best_idx} with score {all_scores[best_idx]}")
            task_variant = self.model_copy(deep=True)
            task_variant.desc = best_prompt
            return task_variant
        else:
            _log("Optimization failed")
            return self


# TODO: Suggest prompts based on successful and unsuccessful examples
async def get_prompts(task: Task, num_prompts: int = 5) -> list[str]:
    optim_task = Task(
        desc=(
            "An LLM needs to be prompted to accomplish the described task. "
            "The LLM's output has to fulfill all of the rules provided. "
            "Suggest prompts in natural language for this LLM to maximize its chances of success. "
            "Each prompt will be shown after all of the input/output schemas are displayed. "
            "Use the correct input/output names in the prompts where relevant. "
            "Do NOT include instructions about JSON schema formats. "
            "Minimize the number of words in the prompts. "
            "Highlight if the task cannot be completed due to contradicting rules or incomplete task descriptions. "
        ),
        inputs=[
            Field(
                name="task_description",
                type="str",
            ),
            Field(
                name="input_schemas",
                type="list",
                list_child_type=Field(
                    name="schema",
                    type="str",
                ),
            ),
            Field(
                name="output_schemas",
                type="list",
                list_child_type=Field(
                    name="schema",
                    type="str",
                ),
            ),
            Field(
                name="rules",
                type="list",
                list_child_type=Field(
                    name="rule",
                    type="str",
                ),
            ),
            Field(
                name="num_prompts",
                type="int",
            ),
        ],
        outputs=[
            Field(
                name="prompts",
                type="list",
                list_child_type=Field(
                    name="prompt",
                    type="str",
                ),
            )
        ],
    )

    max_tries = 3
    for _ in range(max_tries):
        result = await optim_task.run(
            **{
                "task_description": task.desc,
                "input_schemas": [str(i) for i in task.inputs],
                "output_schemas": [str(i) for i in task.outputs],
                "rules": [str(i) for i in task.rules],
                "num_prompts": num_prompts,
            }
        )
        logger.info(result.prompts)
        if not isinstance(result.prompts, str) and len(result.prompts) == num_prompts:
            break

    return result.prompts
