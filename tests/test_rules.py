import otaro.rules as rules

TEST_STRING = "foobar"
TEST_INT = 1


def test_eq():
    assert rules.eq(TEST_INT, TEST_INT) is True
    assert rules.eq(TEST_STRING, TEST_STRING) is True

    assert rules.eq(TEST_INT, TEST_STRING) is False


def test_neq():
    assert rules.neq(TEST_INT, TEST_STRING) is True

    assert rules.neq(TEST_INT, TEST_INT) is False
    assert rules.neq(TEST_STRING, TEST_STRING) is False


def test_contains():
    assert rules.contains([TEST_INT], TEST_INT) is True
    assert rules.contains(TEST_STRING, TEST_STRING[:1]) is True

    assert rules.contains([TEST_INT], TEST_STRING) is False
    assert rules.contains(TEST_STRING, TEST_STRING[::-1]) is False


def test_length_eq():
    assert rules.length_eq([TEST_INT], len([TEST_INT])) is True
    assert rules.length_eq(TEST_STRING, len(TEST_STRING)) is True
