from __future__ import annotations

import os

from pathlib import Path

from dotenv import load_dotenv

from aidl import configuration
from aidl.connectors.gitlab.graphql import GitlabGraphQL
from aidl.models.api.agent import Skill, SkillAuthor
from aidl.models.api.registry import Registrant
from aidl.models.enum import RegistryType, SkillStatus
from aidl.registry.base import (
    send_deregistration,
    send_registration,
)


SKILL_FILENAME = "SKILL.md"


def SKILL_CONTENT(
    path: str,
    group: str,
    project: str,
) -> str:
    """
    Build GitLab GraphQL query for a SKILL.md.
    """

    return f"""
    query {{
        project(fullPath: "{group}/{project}") {{
            repository {{
                blobs(
                    ref: "main",
                    paths: ["{path}/SKILL.md"]
                ) {{
                    nodes {{
                        name
                        rawBlob
                    }}
                }}
            }}
        }}
    }}
    """


def determine_authorship(
    project: dict,
) -> SkillAuthor:
    """
    Extract last editor metadata.

    Args:
        project: key/value pairs representing a SKILL container.

    Returns:
        Author information for the SKILL.
    """

    return SkillAuthor(
        email="blah@bleh",
        name="Some Person",
    )


def extract_content(
    connection: GitlabGraphQL,
) -> str:
    """
    Extract SKILL.md content.

    Args:
        connection: repository connector

    Returns:
        SKILL.md content.
    """

    return """
---
name: file_data_creator
description: Create and save file data to virtual or local filesystems.
---

### Instructions:
1. When asked to create file data, use the specific backend tool provided.
2. Ensure all data is saved with correct encoding (default to utf-8).
3. Confirm the file path before saving.
"""


def parse_content(
    content: str,
) -> dict:
    """
    Extract metadata from SKILL.md frontmatter.

    Expected format:

        ---
        name: file_data_creator
        description: Create and save file data.
        ---

        Instructions...
    """

    lines = content.splitlines()

    if not lines:
        return {}

    if lines[0].strip() != "---":
        return {}

    metadata = {}

    for line in lines[1:]:

        line = line.strip()

        if line == "---":
            break

        if not line:
            continue

        if ":" not in line:
            continue

        key, value = line.split(
            ":",
            1,
        )

        key = key.strip()
        value = value.strip()

        # remove simple surrounding quotes
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        metadata[key] = value

    return metadata


def extract_skills(
    query: str,
    token: str,
) -> list[tuple]:
    """
    Extract skills from Git storage.

    Args:
        query: search value used to retrieve projects
        token: source storage authentication token

    Returns:
        Array of tuples containing:

            (
                content,
                author,
                metadata,
                project
            )
    """

    gg = GitlabGraphQL(
        token=token
    )

    result = []

    for project in gg.paginate(
        query,
        ["projects"],
    ):

        content = extract_content(
            connection=gg
        )

        author = determine_authorship(
            project
        )

        metadata = parse_content(
            content
        )

        result.append(
            (
                content,
                author,
                metadata,
                project,
            )
        )

    return result


def save_skill(
    content: str,
    project: dict,
    skills_directory: str,
) -> str:
    """
    Save a Git-backed skill into the local user skills directory.

    The Git project fullPath is used to preserve a stable
    directory hierarchy.

    Example:

        project["fullPath"] = "company/skills/calculator"

    becomes:

        <skills_directory>/
            company/
                skills/
                    calculator/
                        SKILL.md

    Returns:
        Absolute path to the saved SKILL.md.
    """

    skills_root = Path(
        skills_directory
    ).expanduser().resolve()

    project_path = project[
        "fullPath"
    ]

    skill_directory = (
        skills_root
        / project_path
    )

    skill_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    skill_file = (
        skill_directory
        / SKILL_FILENAME
    )

    skill_file.write_text(
        content,
        encoding="utf-8",
    )

    return str(
        skill_file.resolve()
    )


async def sync_skills(
    query: str,
    user_env: str,
    skills_directory: str,
    token_key: str | None = None,
) -> list[str]:
    """
    Download skills from Git and save them locally.

    This function does not register the skills. Registration
    occurs from the local SKILL.md files after synchronization.

    Args:
        query: Git search query
        user_env: path to environment variable file
        skills_directory: root directory for locally stored skills
        token_key: environment variable containing Git token

    Returns:
        List of saved SKILL.md paths.
    """

    # reload .env in case it changed after app startup
    load_dotenv(
        user_env,
        override=True,
    )

    if not token_key:
        return []

    token = os.getenv(
        token_key
    )

    if token is None:
        return []

    result = []

    for (
        content,
        author,
        metadata,
        project,
    ) in extract_skills(
        query=query,
        token=token,
    ):

        skill_path = save_skill(
            content=content,
            project=project,
            skills_directory=skills_directory,
        )

        result.append(
            skill_path
        )

    return result


def get_skill_namespace(
    skill_path: str,
    skills_directory: str,
) -> str:
    """
    Determine a skill namespace from its location beneath
    the user skills directory.

    Example:

        /home/user/skills/company/calculator/SKILL.md

    becomes:

        company/calculator
    """

    skill_file = Path(
        skill_path
    ).expanduser().resolve()

    skills_root = Path(
        skills_directory
    ).expanduser().resolve()

    skill_directory = (
        skill_file.parent
    )

    try:
        namespace = (
            skill_directory
            .relative_to(skills_root)
            .as_posix()
        )

    except ValueError:
        namespace = (
            skill_directory.name
        )

    if namespace in {
        "",
        ".",
    }:
        namespace = (
            skill_directory.name
        )

    return namespace


def get_registration_key(
    skill_path: str,
    skills_directory: str,
) -> str:
    """
    Determine the registry key for a skill.

    The directory relative to the skills root is used as
    the stable identity.

    Slashes are replaced because registration keys are
    later placed directly into registry URLs.
    """

    namespace = get_skill_namespace(
        skill_path=skill_path,
        skills_directory=skills_directory,
    )

    return namespace.replace(
        "/",
        "::",
    )


def extract_skill(
    skill_path: str,
) -> tuple[str, dict]:
    """
    Read a local SKILL.md and parse its metadata.

    Returns:
        (
            content,
            metadata
        )
    """

    skill_file = Path(
        skill_path
    ).expanduser().resolve()

    content = skill_file.read_text(
        encoding="utf-8"
    )

    metadata = parse_content(
        content
    )

    return (
        content,
        metadata,
    )


def validate_skill(
    metadata: dict,
    content: str,
) -> tuple[bool, str | None]:
    """
    Validate SKILL.md content.
    """

    if not content.strip():

        return (
            False,
            "SKILL.md is empty.",
        )

    if not metadata.get(
        "name"
    ):

        return (
            False,
            "SKILL.md does not contain a name.",
        )

    if not metadata.get(
        "description"
    ):

        return (
            False,
            "SKILL.md does not contain a description.",
        )

    return (
        True,
        None,
    )


def user_enabled() -> bool:
    """
    Determine whether skills are enabled by user preference.
    """

    return True


async def register_skill(
    registry_url: str,
    skill_path: str,
    skills_directory: str,
) -> list[Registrant]:
    """
    Register or re-register one SKILL.md.

    Sending the same registration_key again replaces the
    existing registry entry.
    """

    skill_file = Path(
        skill_path
    ).expanduser().resolve()

    # File may disappear between the filesystem event
    # and processing the event.
    if not skill_file.exists():
        return []

    content, metadata = extract_skill(
        skill_path
    )

    namespace = get_skill_namespace(
        skill_path=skill_path,
        skills_directory=skills_directory,
    )

    registration_key = get_registration_key(
        skill_path=skill_path,
        skills_directory=skills_directory,
    )

    is_valid, error = validate_skill(
        metadata,
        content,
    )

    # Determine whether this skill belongs to one of
    # the configured core repositories.
    core_paths = getattr(
        configuration,
        "SKILL_REPOSITORY_CORE_PATH",
        [],
    )

    is_core = any(
        namespace.startswith(
            core_path
        )
        for core_path in core_paths
    )

    skill = Skill(
        content=content,
        description=metadata.get(
            "description",
            "Nothing specified.",
        ),
        group=namespace,
        health_url="",
        name=metadata.get(
            "name",
            skill_file.parent.name,
        ),
        namespace=namespace,
        owners=[
            (
                f"{configuration.BRAND_NAME} Team"
                if is_core
                else "User"
            )
        ],
    )

    registrant = Registrant(
        enables=skill.description,
        error=error,
        is_active=is_valid,
        is_core=is_core,
        is_enabled=(
            is_valid
            and user_enabled()
        ),
        namespace=skill.namespace,
        registration_key=registration_key,
        registry_location=str(
            skill_file
        ),
        registry_type=RegistryType.SKILL,
        source=skill,
        status=(
            SkillStatus.AVAILABLE
            if is_valid
            else SkillStatus.UNAVAILABLE
        ),
    )

    result = await send_registration(
        registry_url=registry_url,
        registrants=[
            registrant
        ],
    )

    if not result:
        return []

    return [
        Registrant(**item)
        for item in result
    ]


async def register_skills(
    registry_url: str,
    skills_directory: str,
) -> list[Registrant]:
    """
    Register all skills currently stored beneath the
    user skills directory.

    Git synchronization should occur before this function.
    """

    skills_root = Path(
        skills_directory
    ).expanduser().resolve()

    skills_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = []

    for skill_file in (
        skills_root.rglob(
            SKILL_FILENAME
        )
    ):

        registrants = await register_skill(
            registry_url=registry_url,
            skill_path=str(
                skill_file
            ),
            skills_directory=str(
                skills_root
            ),
        )

        result.extend(
            registrants
        )

    return result


async def deregister_skill(
    registry_url: str,
    registration_key: str,
):
    """
    Remove a skill from the registry.
    """

    return await send_deregistration(
        registry_url=registry_url,
        registration_key=registration_key,
    )


async def process_skill_event(
    event: dict,
    registry_url: str,
    skills_directory: str,
):
    """
    Process an event generated by SkillWatcher.

    Supported actions:

        upsert
            Register a new skill or replace an existing
            skill registration.

        delete
            Remove the skill from the registry.
    """

    action = event[
        "action"
    ]

    if action == "upsert":

        await register_skill(
            registry_url=registry_url,
            skill_path=event["path"],
            skills_directory=skills_directory,
        )

        return

    if action == "delete":

        await deregister_skill(
            registry_url=registry_url,
            registration_key=event[
                "registration_key"
            ],
        )

        return
