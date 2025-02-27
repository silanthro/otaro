from otaro import Field


def test_simple_field_attributes():
    field = Field(
        name="foo",
        type=str,
        desc="Hello, world",
    )
    assert field.model == str
    assert field.dummy_value == "Foo"
    assert str(field) == "`foo` (str): Hello, world."
