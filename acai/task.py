import importlib
import inspect
import logging
import re
from pathlib import Path
from typing import Callable

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
    demos: list[dict] = []
    rules: list[Callable[[dict, dict], int | bool]] = []
    config_file: str = ""

    def __init__(
        self,
        desc: str = "",
        inputs: list[Field] = None,
        outputs: list[Field] = None,
        demos: list[dict] = None,
        rules: list[Rule] = None,
        config_file: str = "",
        **kwargs,
    ):
        super().__init__(
            desc=desc,
            inputs=inputs or [],
            outputs=outputs or [],
            demos=demos or [],
            rules=rules or [],
            config_file=config_file,
        )

    @classmethod
    def from_config(cls, config_file: str | Path):
        config_file = Path(config_file)
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
        rules = []
        module = importlib.import_module("acai.rules")
        for rule in config.get("rules", []):
            fn_pattern = re.compile("^(?P<module>.*?)\\.(?P<fn>.*?)$")
            match = fn_pattern.match(rule.strip())
            if match:
                module_name = match.groupdict().get("module")
                fn_name = match.groupdict().get("fn")
                module = importlib.import_module(module_name)
                fn = getattr(module, fn_name)
                rules.append(fn)
            # fn_pattern = re.compile("acai\\.rules\\.(?P<fn>.*?)\\(?P<args>.*?)")
            # match = fn_pattern.match(rule)
            # if match:
            #     module = importlib.import_module("acai.rules")
            #     fn_name = match.groupdict().get("fn")
            #     fn = getattr(module, fn_name)
            #     args_str = match.groupdict().get("args")
            #     # TODO: Better args/kwargs parsing
        config["rules"] = rules

        optimized_config_file = config_file.parent / (config_file.stem + ".optim.yaml")
        optimized_config = {}
        if optimized_config_file.exists():
            optimized_config = cls.from_config(optimized_config_file).model_dump(
                mode="json"
            )
            del optimized_config["config_file"]
            for attr in list(optimized_config.keys()):
                if not optimized_config[attr]:
                    del optimized_config[attr]
        config.update(optimized_config)
        logger.info(f"Config: {config}")
        return cls(config_file=str(config_file), **config)

    @property
    def formatted_desc(self):
        default_desc = f"Given the field{'s' if len(self.inputs) > 1 else ''} {', '.join(f.name for f in self.inputs)}, produce the field{'s' if len(self.outputs) > 1 else ''} {', '.join(f.name for f in self.outputs)}"
        if self.desc:
            return f"{self.desc} {default_desc}"
        else:
            return default_desc

    @property
    def prompt_template(self):
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
            input_values="\n\n".join(field.dummy_template for field in self.inputs),
            output_fields=", ".join(["reasoning"] + [f.name for f in self.outputs]),
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        for demo_kwargs in self.demos:
            demo_user_prompt = USER_PROMPT.format(
                input_values="\n\n".join(
                    field.template(demo_kwargs[field.name]) for field in self.inputs
                ),
                output_fields=", ".join(["reasoning"] + [f.name for f in self.outputs]),
            )
            demo_assistant_prompt = (
                "\n\n".join(
                    field.template(demo_kwargs[field.name]) for field in self.outputs
                )
                + "\n\n[[ ## completed ## ]]\n"
            )
            if "reasoning" in demo_kwargs:
                demo_assistant_prompt = (
                    f"[[ ## reasoning ## ]]\n{demo_kwargs['reasoning']}\n\n"
                    + demo_assistant_prompt
                )
            messages.append({"role": "user", "content": demo_user_prompt})
            messages.append({"role": "assistant", "content": demo_assistant_prompt})

        messages.append({"role": "user", "content": user_prompt})

        return {"messages": messages}

    def get_prompt(self, **kwargs):
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
        ]

        for demo_kwargs in self.demos:
            demo_user_prompt = USER_PROMPT.format(
                input_values="\n\n".join(
                    field.template(demo_kwargs[field.name]) for field in self.inputs
                ),
                output_fields=", ".join(["reasoning"] + [f.name for f in self.outputs]),
            )
            demo_assistant_prompt = (
                "\n\n".join(
                    field.template(demo_kwargs[field.name]) for field in self.outputs
                )
                + "\n\n[[ ## completed ## ]]\n"
            )
            if "reasoning" in demo_kwargs:
                demo_assistant_prompt = (
                    f"[[ ## reasoning ## ]]\n{demo_kwargs['reasoning']}\n\n"
                    + demo_assistant_prompt
                )
            messages.append({"role": "user", "content": demo_user_prompt})
            messages.append({"role": "assistant", "content": demo_assistant_prompt})

        messages.append({"role": "user", "content": user_prompt})

        return {"messages": messages}

    async def run(self, optimize=True, **kwargs):
        messages = self.get_prompt(**kwargs).get("messages")

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
        inputs = kwargs
        outputs = groupdict
        # Optimize against rules
        if optimize and self.config_file and len(self.rules):
            logger.info("Evaluating rules")
            for i, rule in enumerate(self.rules, 1):
                outcome = rule(inputs, outputs)
                logger.info(f"Rule {i}: {rule.__name__} - {outcome}")
                evals[i - 1] = outcome
            if not all(evals):
                new_task = await self.optimize([kwargs], num_prompts=1)
                # TODO: Export more optimized params
                config_file = Path(self.config_file)
                optimized_config_file = config_file.parent / (
                    config_file.stem + ".optim.yaml"
                )
                with open(optimized_config_file, "w") as file:
                    yaml.safe_dump(
                        {
                            "desc": new_task.desc,
                        },
                        file,
                        default_flow_style=False,
                    )
                result = await new_task.run(**kwargs, optimize=False)

        return result

    async def evaluate(self, data: list):
        total_score = 0
        num_evals = 0
        for sample in tqdm(data):
            try:
                result = await self.run(**sample, optimize=False)
                rule_evals = []
                for rule in self.rules:
                    rule_evals.append(rule(sample, result.model_dump(mode="json")))
                if all(rule_evals):
                    total_score += 1.0
            except Exception as e:
                logger.info(e)
            num_evals += 1.0
        return total_score / num_evals

    async def optimize(self, data: list, say=None, num_prompts=5):
        def _log(message: str):
            if say:
                say(message)
            logger.info(message)

        _log("Optimizing task...")

        # Run initial evaluation with rules to determine success rate
        initial_score = await self.evaluate(data)
        _log(f"Initial score: {initial_score}")

        # Get prompts
        prompts = await get_prompts(self, num_prompts=num_prompts)

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
        kwargs = {
            "task_description": task.desc,
            "input_schemas": [str(i) for i in task.inputs],
            "output_schemas": [str(i) for i in task.outputs],
            "rules": [inspect.getsource(i) for i in task.rules],
            "num_prompts": num_prompts,
        }
        result = await optim_task.run(**kwargs, optimize=False)
        logger.info(kwargs)
        logger.info(result.prompts)
        if not isinstance(result.prompts, str) and len(result.prompts) == num_prompts:
            break

    return result.prompts
