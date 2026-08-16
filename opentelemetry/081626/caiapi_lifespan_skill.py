from __future__ import annotations

import asyncio

from aidl.filesystem.skill import SkillWatcher
from aidl.registry.skill import (
    process_local_skill_event,
    register_existing_local_skills,
)

from caiapi.lifespan.task import handle_task_error


async def startup(
    registry_url: str,
    skills_directory: str,
) -> tuple[SkillWatcher, asyncio.Task]:
    """
    Start local skill discovery and filesystem monitoring.

    Existing skills are registered at startup.

    Future SKILL.md creation, modification, movement,
    and deletion are handled automatically.
    """

    queue = asyncio.Queue()

    loop = asyncio.get_running_loop()

    watcher = SkillWatcher(
        skills_directory=skills_directory,
        queue=queue,
        loop=loop,
    )

    # Start the async queue consumer.
    task = asyncio.create_task(
        watcher.watch(
            process_local_skill_event,
            registry_url=registry_url,
            skills_directory=skills_directory,
        ),
        name="skill-watch-handler",
    )

    task.add_done_callback(
        handle_task_error
    )

    # Start filesystem watching BEFORE the initial scan.
    #
    # This prevents a small race where somebody could add a
    # skill between the initial scan and watcher startup.
    watcher.start()

    # Register anything already present.
    await register_existing_local_skills(
        registry_url=registry_url,
        skills_directory=skills_directory,
    )

    return (
        watcher,
        task,
    )


async def shutdown(
    watcher: SkillWatcher,
    task: asyncio.Task,
):
    """
    Shut down local skill filesystem monitoring.
    """

    watcher.stop()

    task.cancel()

    try:
        await task

    except asyncio.CancelledError:
        pass
