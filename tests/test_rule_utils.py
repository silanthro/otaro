import pytest
from pydantic import BaseModel

from otaro.rule_utils import eval_rule_str


class SampleClass(BaseModel):
    foo: str
    a_list: list


SAMPLE = SampleClass(foo="bar", a_list=[1, 2, 3])


def test_eval_rule_str_basic():
    rule_eq_int = "otaro.rules.eq(1, 1)"
    assert eval_rule_str(rule_eq_int) is True
    rule_eq_int = "otaro.rules.eq('foo', 'foo')"
    assert eval_rule_str(rule_eq_int) is True


def eval_sample(rule_str: str):
    return eval_rule_str(
        rule_str=rule_str,
        sample=SAMPLE,
    )


def test_eval_rule_str_sample():
    # Basic equality
    assert eval_sample(f"otaro.rules.eq(foo, '{SAMPLE.foo}')") is True
    # Contains
    assert eval_sample(f"otaro.rules.contains(a_list, {SAMPLE.a_list[0]})") is True
    # Check length
    assert eval_sample(f"otaro.rules.length_eq(a_list, {len(SAMPLE.a_list)})") is True
    # Check list parsing
    assert eval_sample(f"otaro.rules.contains(['{SAMPLE.foo}'], foo)") is True
    assert eval_sample(f"otaro.rules.contains(['{SAMPLE.foo}',], foo)") is True
    assert (
        eval_sample(f"otaro.rules.contains(['{SAMPLE.foo}','{SAMPLE.foo}'], foo)")
        is True
    )
    # Check tuple parsing
    assert eval_sample(f"otaro.rules.contains(('{SAMPLE.foo}'), foo)") is True
    assert eval_sample(f"otaro.rules.contains(('{SAMPLE.foo}',), foo)") is True
    assert (
        eval_sample(f"otaro.rules.contains(('{SAMPLE.foo}','{SAMPLE.foo}'), foo)")
        is True
    )


def test_eval_rule_str_kwargs():
    # Test kwargs
    assert eval_sample(f"otaro.rules.eq(foo, b='{SAMPLE.foo}')")
    assert eval_sample(f"otaro.rules.eq(a=foo, b='{SAMPLE.foo}')")


def test_eval_rule_str_no_args():
    # Test rules with no args
    assert eval_sample("otaro.rules.always_true") is True
    assert eval_sample("otaro.rules.always_false") is False


def test_invalid_rule():
    # Check that error is raised correctly
    invalid_rule = "invalid rule"
    with pytest.raises(ValueError, match="Invalid rule"):
        eval_rule_str(invalid_rule)
