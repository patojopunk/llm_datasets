from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
)

skills_backend = FilesystemBackend(
    root_dir=skills_dir,
    virtual_mode=True,
)

backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/skills/": skills_backend,
    },
)

agent = create_deep_agent(
    ...,
    backend=backend,
    skills=["/skills/"],
)

#---------------------------------------------------------


from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
)

backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/skills/": FilesystemBackend(
            root_dir="/my/home/dir/skills",
            virtual_mode=True,
        ),
    },
)

agent = create_deep_agent(
    model=model,
    backend=backend,
    skills=["/skills/"],
)
