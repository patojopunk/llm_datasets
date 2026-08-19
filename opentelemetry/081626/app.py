@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, None]:

    # STARTUP ========================================

    app.state[configuration.QUEUE_REGISTRY] = asyncio.Queue()
    app.state[configuration.QUEUE_SYSTEM] = asyncio.Queue()

    # data layer
    await datalayer_startup(
        app.state,
        app.state[configuration.QUEUE_REGISTRY],
    )

    # downloads/registers resources and establishes
    # things like the skills directory
    await registry_startup(
        app.state,
        app.state[configuration.QUEUE_REGISTRY],
    )

    # certificate watcher
    cert_watch_task = await filesystem_startup(
        app.state,
        app.state[configuration.QUEUE_SYSTEM],
    )

    # ------------------------------------------------
    # Skill watcher
    # ------------------------------------------------

    skills_directory = app.state[
        "skills_directory"
    ]


    from aidl.models.enum import ServiceFunction
    from aidl.registry.base import get_service
    
    gateway = get_service(
        app.state,
        functions=[
            ServiceFunction.GATEWAY
        ],
    )
    
    registry_url = gateway.registry_location
    
    skills_directory = app.state[
        "skills_directory"
    ]


  
    skill_queue = asyncio.Queue()

    skill_watcher = SkillWatcher(
        skills_directory=skills_directory,
        queue=skill_queue,
        loop=asyncio.get_running_loop(),
    )

    skill_watch_task = asyncio.create_task(
        skill_watcher.watch(
            process_skill_event,
            registry_url=registry_url,
            skills_directory=skills_directory,
        ),
        name="skill-watch-handler",
    )

    skill_watch_task.add_done_callback(
        handle_task_error
    )

    skill_watcher.start()

    # registry listener
    registry_event_task = asyncio.create_task(
        listen_to_registry(
            storage=app.state,
            registry_queue=app.state[
                configuration.QUEUE_REGISTRY
            ],
            system_queue=app.state[
                configuration.QUEUE_SYSTEM
            ],
        ),
        name="registry-event-handler",
    )

    registry_event_task.add_done_callback(
        handle_task_error
    )

    yield

    # SHUTDOWN ========================================

    skill_watcher.stop()

    await task_shutdown(skill_watch_task)
    await task_shutdown(cert_watch_task)
    await task_shutdown(registry_event_task)
