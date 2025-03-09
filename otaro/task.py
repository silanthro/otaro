import asyncio
import json
import logging
import re
from pathlib import Path

import yaml
from litellm import acompletion, token_counter
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from tqdm import tqdm

from otaro.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from otaro.rule_utils import eval_rule_str, get_rule_source
from otaro.task_utils import parse_custom_types, parse_fields_config
from otaro.types import Field, FieldParsingError

logging.basicConfig()
logger = logging.getLogger("otaro.task")
logger.setLevel(logging.INFO)


class CompletionResponse(BaseModel):
    content: str
    num_input_tokens: int
    num_output_tokens: int


class CommonError(BaseModel):
    field: str
    error_message: str
    correct_dummy_sample: str


def count_tokens(
    message: str,
    role="user",
    model: str | None = None,
):
    return token_counter(
        model=model,
        messages=[
            {
                "role": role,
                "content": message,
            }
        ],
    )


async def _completion(
    model: str,
    messages: list[dict],
):
    model = model
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
    model: str
    inputs: list[Field]
    outputs: list[Field]
    desc: str = ""
    demos: list[dict] = []
    rules: list[str] = []
    config_file: str | None = None

    def __init__(
        self,
        model: str,
        inputs: list[Field | str],
        outputs: list[Field | str],
        desc: str = "",
        demos: list[dict] | None = None,
        rules: list[str] | None = None,
        config_file: str | None = None,
        **kwargs,
    ):
        inputs = self.parse_fields(inputs)
        outputs = self.parse_fields(outputs)
        super().__init__(
            model=model,
            inputs=inputs or [],
            outputs=outputs or [],
            desc=desc,
            demos=demos or [],
            rules=rules or [],
            config_file=config_file,
        )

    @staticmethod
    def parse_fields(fields: list[Field | str] | None = None):
        fields = fields or []
        parsed_fields = []
        for field in fields:
            if isinstance(field, str):
                field = Field(name=field)
            elif isinstance(field, Field):
                pass
            else:
                Field(**field)
            parsed_fields.append(field)
        return parsed_fields

    # TODO: Check for duplicate field name between inputs and outputs
    @classmethod
    def from_config(cls, config_file: str | Path):
        config_file = Path(config_file)
        with open(config_file) as file:
            config = yaml.safe_load(file)
        custom_types = {}
        if config.get("custom_types"):
            custom_types = parse_custom_types(config.get("custom_types"))
            del config["custom_types"]
        config["inputs"] = parse_fields_config(
            config.get("inputs", []), custom_types=custom_types
        )
        config["outputs"] = parse_fields_config(
            config.get("outputs", []), custom_types=custom_types
        )

        optimized_config_file = config_file.parent / (config_file.stem + ".optim.yml")
        optimized_config = {}
        if optimized_config_file.exists():
            with open(optimized_config_file) as file:
                optimized_config = yaml.safe_load(file) or {}
            if optimized_config.get("inputs"):
                optimized_config["inputs"] = parse_fields_config(
                    optimized_config.get("inputs", []), custom_types=custom_types
                )
            if optimized_config.get("outputs"):
                optimized_config["outputs"] = parse_fields_config(
                    optimized_config.get("outputs", []), custom_types=custom_types
                )

            # optimized_config = cls.from_config(optimized_config_file).model_dump(
            #     mode="json"
            # )
            # del optimized_config["config_file"]
            # for attr in list(optimized_config.keys()):
            #     if not optimized_config[attr]:
            #         del optimized_config[attr]
        config.update(optimized_config)
        # logger.info(f"Config: {config}")
        return cls(config_file=str(config_file), **config)

    @property
    def formatted_desc(self):
        default_desc = f"Given the field{'s' if len(self.inputs) > 1 else ''} {', '.join(f.name for f in self.inputs)}, produce the field{'s' if len(self.outputs) > 1 else ''} {', '.join(f.name for f in self.outputs)}"
        if self.desc:
            return f"{self.desc} {default_desc}"
        else:
            return default_desc

    @property
    def dummy_input(self):
        dummy_inputs = {}
        for input_field in self.inputs:
            dummy_inputs[input_field.name] = input_field.dummy_value
        return dummy_inputs

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
            task=self.desc,
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
                task=self.desc,
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
            task=self.desc,
            output_fields=", ".join(["reasoning"] + [f.name for f in self.outputs]),
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        for demo_kwargs in self.demos:
            for field in self.inputs:
                if demo_kwargs.get(field.name) is None:
                    demo_kwargs[field.name] = field.default
            demo_user_prompt = USER_PROMPT.format(
                input_values="\n\n".join(
                    field.template(demo_kwargs[field.name]) for field in self.inputs
                ),
                task=self.desc,
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

    def run(self, optimize=True, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.arun(optimize, **kwargs))

    def create_optim_file(self, optimized_task: "Task"):
        if not self.config_file:
            raise ValueError(
                "Cannot export optimized config for task without original config file"
            )
        # TODO: Export more optimized params
        config_file = Path(self.config_file)
        optimized_config_file = config_file.parent / (config_file.stem + ".optim.yml")
        inputs_config = []
        for field in optimized_task.inputs:
            field_json = field.model_dump(
                mode="python",
                exclude=["type"],
                exclude_defaults=True,
                exclude_none=True,
                exclude_unset=True,
            )
            field_json["type"] = field.get_type_as_dict()
            inputs_config.append(field_json)
        outputs_config = []
        for field in optimized_task.outputs:
            field_json = field.model_dump(
                mode="python",
                exclude=["type"],
                exclude_defaults=True,
                exclude_none=True,
                exclude_unset=True,
            )
            field_json["type"] = field.get_type_as_dict()
            outputs_config.append(field_json)
        with open(optimized_config_file, "w") as file:
            yaml.safe_dump(
                {
                    "desc": optimized_task.desc,
                    "inputs": inputs_config,
                    "outputs": outputs_config,
                    # "inputs": [
                    #     i.model_dump(
                    #         mode="python",
                    #         exclude_defaults=True,
                    #         exclude_none=True,
                    #         exclude_unset=True,
                    #     )
                    #     for i in optimized_task.inputs
                    # ],
                },
                file,
                default_flow_style=False,
            )

    async def arun(self, optimize=True, verbose=True, **kwargs):
        # TODO: Handle edge case where None is actually intended
        for field in self.inputs:
            if kwargs.get(field.name) is None:
                kwargs[field.name] = field.default

        messages = self.get_prompt(**kwargs).get("messages")
        result = None

        num_tries = 3

        for _ in range(num_tries):
            response = await _completion(model=self.model, messages=messages)

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
                logger.warning(f"No match on {response.content}")
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
                    matched_groups = match.groupdict()
                    # logger.info(messages)
                    # logger.info(matched_groups["reasoning"])
                    # logger.info(matched_groups["tweets"])
                    groupdict = {
                        "num_input_tokens": response.num_input_tokens,
                        "num_output_tokens": response.num_output_tokens,
                    }
                    while True:
                        try:
                            for k, v in matched_groups.items():
                                if k == "reasoning":
                                    groupdict[k] = str(v.strip())
                                else:
                                    output_field = [
                                        o for o in self.outputs if o.name == k
                                    ][0]
                                    groupdict[k] = output_field.parse(
                                        v.strip(), to_dict=True
                                    )
                            result = model(**groupdict, **kwargs)
                            break
                        except FieldParsingError as e:
                            logger.exception(e)
                            logger.info("Correcting error...")
                            # Get corrected field
                            correction = await get_field_correction(
                                field=e.field,
                                model_response=match.groupdict()[e.field.name],
                                error_message=str(e),
                                model=self.model,
                            )
                            # Get error idx
                            error_idx = 0
                            for field in self.inputs:
                                if field.name.startswith(
                                    f"previous_error_example_{e.field.name}"
                                ):
                                    error_idx += 1
                            common_error_field = (
                                f"previous_error_example_{e.field.name}_{error_idx}"
                            )
                            self.inputs.append(
                                Field(
                                    name=common_error_field,
                                    desc="A previous error from another response.",
                                    type=CommonError,
                                    default={
                                        "field": e.field.name,
                                        "error_message": correction.dummy_error_message,
                                        "correct_dummy_sample": json.dumps(
                                            e.field.dummy_value
                                        ),
                                    },
                                )
                            )
                            logger.info("Completed error correction")
                            # TODO: Support optim export for common errors
                            self.create_optim_file(self)
                            matched_groups[e.field.name] = correction.correct_output

                    # Optimize against rules
                    evals = [False] * len(self.rules)
                    if len(self.rules):
                        if verbose:
                            logger.info("Evaluating rules")
                        for i, rule in enumerate(self.rules, 1):
                            outcome = eval_rule_str(rule, result)
                            if verbose:
                                logger.info(f"Rule {i}: {rule} - {outcome}")
                            evals[i - 1] = outcome

                    if not all(evals):
                        # Optimize if possible
                        if optimize and self.config_file:
                            new_task, best_results = await self.optimize(
                                [kwargs], num_prompts=3
                            )
                            self.create_optim_file(new_task)
                            # Update to optimized task
                            self.sync_to(new_task)
                            result = best_results

                    return result
                except Exception as e:
                    logger.exception(e)
                    logger.info("Retrying...")
        raise ValueError("Unable to generate valid response")

    def evaluate(self, data: list, verbose=False):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aevaluate(data, verbose))

    async def aevaluate(self, data: list, verbose=False):
        num_evals = 0
        results = []
        scores = []
        best_score = 0
        best_result = None
        for sample in tqdm(data):
            try:
                result = await self.arun(**sample, optimize=False, verbose=verbose)
                rule_evals = []
                for rule in self.rules:
                    outcome = eval_rule_str(rule, result)
                    rule_evals.append(outcome)
                # Assume rule_evals is list[bool]
                # TODO: Handle other cases
                score = sum(rule_evals) / len(rule_evals)
                scores.append(score)
                results.append(result)
                if score > best_score:
                    best_result = result
                    best_score = score
            except Exception as e:
                logger.exception(e)
            num_evals += 1.0
        if len(scores) == 0:
            return 0, None
        else:
            return sum(scores) / len(scores), best_result

    async def optimize(
        self,
        data: list,
        say=None,
        num_prompts=5,
        trials_per_prompt=1,
        terminate_asap=True,
    ):
        def _log(message: str):
            if say:
                say(message)
            logger.info(message)

        _log("Optimizing task...")

        # Run initial evaluation with rules to determine success rate
        initial_score, best_initial_result = await self.aevaluate(
            data * trials_per_prompt
        )
        _log(f"Initial score: {initial_score}")

        # Get prompts
        suggestions = await get_prompts(self, num_prompts=num_prompts)

        # For each prompt, run evaluation
        # TODO: Add prompt-demo combination
        variant_scores = []
        variant_results = []
        variants = []
        for prompt in suggestions.task_prompts:
            _log(f"Evaluating prompt: {prompt}")
            task_variant = self.model_copy(deep=True)
            task_variant.desc = prompt
            for field in task_variant.outputs:
                field.desc = suggestions.model_dump().get(f"{field.name}_desc_rules")
            score, best_result = await task_variant.aevaluate(data * trials_per_prompt)
            variant_scores.append(score)
            variant_results.append(best_result)
            variants.append(task_variant)
            _log(f"\tScore: {score}")
            _log(f"All scores: {[initial_score] + variant_scores}")
            # Terminate once a prompt that fulfills all rules is found
            if terminate_asap and score == 1.0:
                break

        # Select best parameters
        if max(variant_scores) >= initial_score:
            all_scores = [initial_score] + variant_scores
            all_results = [best_initial_result] + variant_results
            all_variants = [self] + variants

            # Select best-performing prompt that is the shortest
            best_idxs = [
                i for i, score in enumerate(all_scores) if score >= max(all_scores)
            ]
            best_variants = [all_variants[i] for i in best_idxs]
            prompt_lengths = [count_tokens(t.desc) for t in best_variants]
            shortest_idxs = [
                i
                for i, length in enumerate(prompt_lengths)
                if length <= min(prompt_lengths)
            ]
            best_variant = best_variants[shortest_idxs[0]]
            best_idx = all_variants.index(best_variant)
            _log(f"Selecting prompt #{best_idx} with score {all_scores[best_idx]}")
            return best_variant, all_results[best_idx]
        else:
            _log("Optimization failed")
            return self, best_initial_result

    def sync_to(self, new_task: "Task"):
        # Used to update self to optimized version
        # TODO: Sync more attributes
        self.desc = new_task.desc
        # TODO: Make this more elegant
        for field in new_task.inputs:
            if field.name not in [f.name for f in self.inputs]:
                self.inputs.append(field)
        for field in new_task.outputs:
            if field.name not in [f.name for f in self.outputs]:
                self.outputs.append(field)


# TODO: Suggest prompts based on successful and unsuccessful examples
async def get_prompts(
    task: Task, model: str | None = None, num_prompts: int = 5
) -> BaseModel:
    optim_task = Task(
        model=model or task.model,
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
            "task_description",
            Field(
                name="input_schemas",
                type=list[str],
            ),
            Field(
                name="output_schemas",
                type=list[str],
            ),
            Field(
                name="rules",
                type=list[str],
            ),
            Field(
                name="num_prompts",
                type=int,
            ),
        ],
        outputs=[
            Field(
                name="task_prompts",
                type=list[str],
            ),
            *[
                Field(
                    name=f"{f.name}_desc_rules",
                    desc="A description for the field and any relevant rules. Exclude JSON format.",
                )
                for f in task.outputs
            ],
        ],
    )

    max_tries = 3
    for _ in range(max_tries):
        kwargs = {
            "task_description": task.desc,
            "input_schemas": [str(i) for i in task.inputs],
            "output_schemas": [str(i) for i in task.outputs],
            # TODO: Need to handle source different for rules with and without args
            # Current getsource only works with rules that do not have args i.e. takes in sample
            # For rules with args e.g. otaro.rules.contains(haiku, "green") we probably need to
            # also pass the rule string
            "rules": [get_rule_source(i) for i in task.rules],
            "num_prompts": num_prompts,
        }
        result = await optim_task.arun(**kwargs, optimize=False)
        if (
            not isinstance(result.task_prompts, str)
            and len(result.task_prompts) == num_prompts
        ):
            break
    return result


async def get_field_correction(
    field: Field,
    model_response: str,
    error_message: str,
    model: str | None = None,
):
    correction_task = Task(
        model=model,
        inputs=[
            "output_schema",
            "wrong_output",
            "error_message",
            "correct_dummy_output",
        ],
        outputs=[
            "correct_output",
            Field(
                name="dummy_error_message",
                desc="Rewrite error_message to replace user input info with dummy info",
            ),
        ],
    )

    max_tries = 3
    for _ in range(max_tries):
        kwargs = {
            "output_schema": str(field),
            "wrong_output": model_response,
            "error_message": error_message,
            "correct_dummy_output": json.dumps(field.dummy_value),
        }
        result = await correction_task.arun(**kwargs, optimize=False)

        return result
