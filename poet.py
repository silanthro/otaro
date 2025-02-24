import asyncio

from acai.task import Task


async def main():
    task = Task.from_config("poet.yaml")

    response = await task.run(topic="frogs")

    print(response.haiku)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
