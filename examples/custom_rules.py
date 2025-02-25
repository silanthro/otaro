from pydantic import BaseModel


def haiku_contains_green(sample: BaseModel):
    return "green" in sample.haiku.lower()


def haiku_contains_blue(sample: BaseModel):
    return "blue" in sample.haiku.lower()
