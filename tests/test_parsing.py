import json

import pytest

from otaro.parsing import llm_parse_json

SAMPLE_DICT = {
    "foo": "bar",
    "answer": 42,
    "hello world": [1, 2, 3.14],
    "empty": None,
}
SAMPLE_DICT_STR = json.dumps(SAMPLE_DICT)

SAMPLE_ARRAY = ["foo", 1, [3.14], {"hello": "world"}, None]
SAMPLE_ARRAY_STR = json.dumps(SAMPLE_ARRAY)


def test_llm_parse_json_dict():
    assert llm_parse_json(SAMPLE_DICT_STR) == SAMPLE_DICT


def test_llm_parse_json_array():
    assert llm_parse_json(SAMPLE_ARRAY_STR) == SAMPLE_ARRAY


def test_llm_parse_json_handle_single_quotes():
    assert llm_parse_json(SAMPLE_DICT_STR.replace('"', "'")) == SAMPLE_DICT
    assert llm_parse_json(SAMPLE_ARRAY_STR.replace('"', "'")) == SAMPLE_ARRAY


def test_llm_parse_json_none_string():
    # Handles edge case where LLM produces "None" instead of "null"
    assert llm_parse_json(SAMPLE_DICT_STR.replace("null", "None")) == SAMPLE_DICT
    assert llm_parse_json(SAMPLE_ARRAY_STR.replace("null", "None")) == SAMPLE_ARRAY


def test_llm_parse_json_invalid():
    # Raises error for invalid string
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        llm_parse_json("[invalid]")
