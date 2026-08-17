import asyncio
import httpx

from aidl.registry.base import get_service
from aidl.registry.skill import process_skill_event

from aidl.filesystem.skill import SkillWatcher
from aiidentity.certificate.store import CertificateWatcher

from caiapi.lifespan.registry import (
    TOKEN_SERVICES,
    update_certificate_services,
)
from caiapi.lifespan.task import handle_task_error


async def send_to_event_service(event: dict, **kwargs):
    """
    Process to run on each certificate event message.

    Args:
        event: key/values representing a system event
        endpoint: (kwargs) url from service registration
        queue: (kwargs) messaging queue for system events
        registrant: (kwargs) messaging service represented as a registrant
        storage: (kwargs) registry storage
    """

    queue = kwargs.get("queue")
    registrant = kwargs.get("registrant")
    storage = kwargs.get("storage")

    # send to system service events
    await queue.put(event)

    # revalidate associated services
    await update_certificate_services(
        storage,
        TOKEN_SERVICES,
        queue,
    )

    # send to event broadcast
    if registrant and registrant.is_active:

        url = f"{kwargs.get('endpoint')}/events"

        try:

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=event.model_dump(
                        mode="json"
                    ),
                )

                response.raise_for_status()

        except httpx.RequestError:
            print(
                f"Could not connect to service via {url}"
            )

        except Exception as e:
            print(
                f"General problem sending event via "
                f"{url}: {e}"
            )


async def startup(
    storage: dict,
    queue: asyncio.Queue,
    registry_url: str,
    skills_directory: str,
) -> list[asyncio.Task]:
    """
    Initialize filesystem processes at startup.

    Starts:
        - certificate filesystem watcher
        - skill filesystem watcher
    """

    # attempt to get messaging service
    registrant = get_service(
        storage,
        functions="messaging",
    )

    tasks = []

    # ---------------------------------------------------------
    # Certificate watcher
    # ---------------------------------------------------------

    cw = CertificateWatcher()

    certificate_task = asyncio.create_task(
        cw.watch(
            send_to_event_service,
            endpoint=registrant.registry_location,
            queue=queue,
            registrant=registrant,
            storage=storage,
        ),
        name="certificate-watch-handler",
    )

    certificate_task.add_done_callback(
        handle_task_error
    )

    cw.start(
        recursive=False
    )

    tasks.append(
        certificate_task
    )

    # ---------------------------------------------------------
    # Skill watcher
    # ---------------------------------------------------------

    skill_queue = asyncio.Queue()

    loop = asyncio.get_running_loop()

    sw = SkillWatcher(
        skills_directory=skills_directory,
        queue=skill_queue,
        loop=loop,
    )

    skill_task = asyncio.create_task(
        sw.watch(
            process_skill_event,
            registry_url=registry_url,
            skills_directory=skills_directory,
        ),
        name="skill-watch-handler",
    )

    skill_task.add_done_callback(
        handle_task_error
    )

    sw.start()

    tasks.append(
        skill_task
    )

    return tasks
