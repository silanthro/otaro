import click
import uvicorn
from pydantic import create_model
from pydantic.fields import FieldInfo

from acai.api.server import app
from acai.task import Task


@click.command()
@click.argument("config_file")
def deploy(config_file: str):
    print(config_file)
    task = Task.from_config(config_file)
    input_field_attributes = {}
    for field in task.inputs:
        input_field_attributes[field.name] = (field.model, FieldInfo())
    PayloadClass = create_model(
        "Payload",
        **input_field_attributes,
    )

    async def run(payload: PayloadClass):
        return await task.run(optimize=False, **payload.model_dump(mode="json"))

    app.add_api_route("/run", endpoint=run, methods=["POST"])
    uvicorn.run(app)
