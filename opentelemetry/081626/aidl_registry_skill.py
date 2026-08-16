from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from aidl import configuration
from aidl.connectors.gitlab.graphql import GitlabGraphQL
from aidl.models.api.agent import Skill, SkillAuthor
from aidl.models.api.registry import Registrant
from aidl.models.enum import RegistryType, SkillStatus
from aidl.registry.base import send_registration


def SKILL_CONTENT(
    path: str,
    group: str,
    project: str,
) -> str:
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


def determine_authorship(project: dict) -> SkillAuthor:
    """
    Extract last editor metadata.

    Args:
        project: key/value pairs representing a SKILL container.

    Returns:
        Skill author information.
    """

    return SkillAuthor(
        email="blah@bleh",
        name="Some Person",
    )


def extract_content(
    connection: GitlabGraphQL,
) -> str:
    """
    Extract SKILL.md content from repository storage.

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


def extract_skills(
    query: str,
    token: str,
) -> list[tuple]:
    """
    Extract skills from GitLab storage.

    Returns:
        Tuples containing:

            (
                content,
                author,
                metadata,
                project
            )
    """

    gg = GitlabGraphQL(
        token=token,
    )

    result = []

    for project in gg.paginate(
        query,
        ["projects"],
    ):
        content = extract_content(
            connection=gg,
        )

        author = determine_authorship(
            project,
        )

        metadata = parse_content(
            content,
            project,
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


def parse_content(
    content: str,
    project: dict,
) -> dict:
    """
    Extract metadata from repository SKILL content.
    """

    return {
        "description": project["description"],
        "name": project["name"],
    }


async def register_skills(
    registry_url: str,
    query: str,
    user_env: str,
    token_key: str | None = None,
) -> list[Registrant]:
    """
    Register repository skills with the gateway.
    """

    # Reload .env in case it changed after application startup.
    load_dotenv(
        user_env,
        override=True,
    )

    if not token_key:
        return []

    token = os.getenv(
        token_key,
    )

    if token is None:
        return []

    skills = [
        (
            validate_skill(
                metadata,
                content,
            ),
            user_enabled(),
            Skill(
                content=content,
                description=metadata.get(
                    "description",
                    "Nothing specified.",
                ),
                group=project["fullPath"],
                health_url="fill in from project",
                name=metadata.get(
                    "name",
                    "Unknown",
                ),
                namespace=project["fullPath"],
                owners=[
                    (
                        f"{configuration.BRAND_NAME} Team"
                        if project["fullPath"]
                        in configuration.SKILL_REPOSITORY_CORE_PATH
                        else (
                            author.name
                            if author
                            else "Unknown"
                        )
                    )
                ],
            ),
        )
        for content, author, metadata, project
        in extract_skills(query, token)
    ]

    registrants = await send_registration(
        registry_url=registry_url,
        registrants=[
            Registrant(
                enables=skill.description,
                error=error,
                is_active=is_valid,
                is_core=(
                    skill.group
                    in configuration.SKILL_REPOSITORY_CORE_PATH
                ),
                is_enabled=(
                    is_valid
                    and is_enabled
                ),
                namespace=skill.group,
                registration_key=skill.namespace,
                registry_location=(
                    f"{skill.group}/"
                    f"{skill.name}/"
                    "SKILL.md"
                ),
                registry_type=RegistryType.SKILL,
                source=skill,
                status=(
                    SkillStatus.AVAILABLE
                    if is_valid
                    else SkillStatus.UNAVAILABLE
                ),
            )
            for (
                (is_valid, error),
                is_enabled,
                skill,
            )
            in skills
        ],
    )

    return [
        Registrant(**registrant)
        for registrant in registrants
    ]


# -------------------------------------------------------------------
# Local user skills
# -------------------------------------------------------------------


def parse_local_skill_content(
    content: str,
) -> dict:
    """
    Parse the simple YAML frontmatter from a local SKILL.md.

    Expected structure:

        ---
        name: calculator
        description: Perform calculations.
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

        if line.startswith("#"):
            continue

        if ":" not in line:
            continue

        key, value = line.split(
            ":",
            1,
        )

        key = key.strip()
        value = value.strip()

        # Remove simple surrounding quotes.
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        metadata[key] = value

    return metadata


def get_local_skill_identity(
    skill_path: str,
    skills_directory: str,
) -> tuple[str, str, str]:
    """
    Determine the stable registry identity for a local skill.

    Returns:
        (
            registration_key,
            namespace,
            default_name
        )
    """

    skill_file = Path(
        skill_path
    ).expanduser().resolve()

    skills_root = Path(
        skills_directory
    ).expanduser().resolve()

    skill_directory = skill_file.parent

    try:
        relative_directory = skill_directory.relative_to(
            skills_root
        )

        namespace = relative_directory.as_posix()

    except ValueError:
        # Fallback if somebody passes a SKILL.md outside
        # of the configured root.
        namespace = skill_directory.name

    if namespace in {"", "."}:
        namespace = skill_directory.name

    default_name = skill_directory.name

    # Prefix local skills so they do not collide with
    # repository-backed registration keys.
    registration_key = f"local:{namespace}"

    return (
        registration_key,
        namespace,
        default_name,
    )


def extract_local_skill(
    skill_path: str,
) -> tuple[str, dict]:
    """
    Read a local SKILL.md and extract its metadata.
    """

    skill_file = Path(
        skill_path
    ).expanduser().resolve()

    content = skill_file.read_text(
        encoding="utf-8",
    )

    metadata = parse_local_skill_content(
        content,
    )

    return (
        content,
        metadata,
    )


def build_local_skill(
    skill_path: str,
    skills_directory: str,
) -> tuple[
    tuple[bool, str | None],
    Registrant,
]:
    """
    Build a registry object for a local skill.
    """

    content, metadata = extract_local_skill(
        skill_path,
    )

    registration_key, namespace, default_name = (
        get_local_skill_identity(
            skill_path,
            skills_directory,
        )
    )

    is_valid, error = validate_skill(
        metadata,
        content,
    )

    skill = Skill(
        content=content,
        description=metadata.get(
            "description",
            "Nothing specified.",
        ),
        group="local",
        health_url="",
        name=metadata.get(
            "name",
            default_name,
        ),
        namespace=namespace,
        owners=["User"],
    )

    registrant = Registrant(
        enables=skill.description,
        error=error,
        is_active=is_valid,
        is_core=False,
        is_enabled=(
            is_valid
            and user_enabled()
        ),
        namespace=namespace,
        registration_key=registration_key,
        registry_location=str(
            Path(skill_path).expanduser().resolve()
        ),
        registry_type=RegistryType.SKILL,
        source=skill,
        status=(
            SkillStatus.AVAILABLE
            if is_valid
            else SkillStatus.UNAVAILABLE
        ),
    )

    return (
        (is_valid, error),
        registrant,
    )


async def register_local_skill(
    registry_url: str,
    skill_path: str,
    skills_directory: str,
) -> list[Registrant]:
    """
    Register or re-register a local SKILL.md.

    Sending the same registration_key again is expected
    to update the existing registry entry.
    """

    skill_file = Path(
        skill_path
    ).expanduser().resolve()

    # A delete could occur between receiving the filesystem
    # event and processing it.
    if not skill_file.exists():
        return await register_deleted_local_skill(
            registry_url=registry_url,
            skill_path=skill_path,
            skills_directory=skills_directory,
        )

    _, registrant = build_local_skill(
        skill_path=skill_path,
        skills_directory=skills_directory,
    )

    result = await send_registration(
        registry_url=registry_url,
        registrants=[
            registrant,
        ],
    )

    return [
        Registrant(**item)
        for item in result
    ]


async def register_deleted_local_skill(
    registry_url: str,
    skill_path: str,
    skills_directory: str,
) -> list[Registrant]:
    """
    Re-register a deleted local skill as unavailable.

    The important part is that this uses the exact same
    registration_key that the skill used before deletion.
    """

    registration_key, namespace, default_name = (
        get_local_skill_identity(
            skill_path,
            skills_directory,
        )
    )

    skill = Skill(
        content="Local skill has been deleted.",
        description="Local skill has been deleted.",
        group="local",
        health_url="",
        name=default_name,
        namespace=namespace,
        owners=["User"],
    )

    registrant = Registrant(
        enables=skill.description,
        error="SKILL.md was deleted from the local filesystem.",
        is_active=False,
        is_core=False,
        is_enabled=False,
        namespace=namespace,
        registration_key=registration_key,
        registry_location=str(
            Path(skill_path).expanduser().resolve()
        ),
        registry_type=RegistryType.SKILL,
        source=skill,
        status=SkillStatus.UNAVAILABLE,
    )

    result = await send_registration(
        registry_url=registry_url,
        registrants=[
            registrant,
        ],
    )

    return [
        Registrant(**item)
        for item in result
    ]


async def register_existing_local_skills(
    registry_url: str,
    skills_directory: str,
) -> list[Registrant]:
    """
    Register every local skill already present at startup.
    """

    skills_root = Path(
        skills_directory
    ).expanduser().resolve()

    skills_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result: list[Registrant] = []

    for skill_file in skills_root.rglob(
        "SKILL.md"
    ):
        registrants = await register_local_skill(
            registry_url=registry_url,
            skill_path=str(skill_file),
            skills_directory=str(skills_root),
        )

        result.extend(
            registrants
        )

    return result


async def process_local_skill_event(
    event: dict,
    registry_url: str,
    skills_directory: str,
):
    """
    Process an event generated by SkillWatcher.
    """

    action = event["action"]
    path = event["path"]

    if action == "upsert":

        await register_local_skill(
            registry_url=registry_url,
            skill_path=path,
            skills_directory=skills_directory,
        )

        return

    if action == "delete":

        await register_deleted_local_skill(
            registry_url=registry_url,
            skill_path=path,
            skills_directory=skills_directory,
        )


def validate_skill(
    metadata: dict,
    content: str,
) -> tuple[bool, str | None]:
    """
    Basic SKILL.md validation.
    """

    if not content.strip():
        return (
            False,
            "SKILL.md is empty.",
        )

    if not metadata.get("name"):
        return (
            False,
            "SKILL.md does not contain a name.",
        )

    if not metadata.get("description"):
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
    Determine whether user skills are enabled.
    """

    return True
