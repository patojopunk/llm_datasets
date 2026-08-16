from __future__ import annotations

import asyncio
import time
from pathlib import Path

from watchdog.events import FileSystemEvent

from aidl.filesystem.directory import DirectoryHandler, DirectoryWatcher


class SkillHandler(DirectoryHandler):
    """
    Watch for changes to SKILL.md files inside a skills directory.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        skills_directory: str,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__(queue=queue)

        self.skills_directory = Path(skills_directory).expanduser().resolve()
        self.loop = loop

        # Track existing skills so directory deletes/moves can also
        # be associated with their SKILL.md files.
        self.known_skill_files = {
            str(path.resolve())
            for path in self.skills_directory.rglob("SKILL.md")
        }

        # Filesystem watchers can emit several events for one save.
        # This prevents very rapid duplicate registrations.
        self._last_events: dict[tuple[str, str], float] = {}
        self._debounce_seconds = 0.25

    def _is_skill_file(self, path: str | Path) -> bool:
        """
        Determine whether a path points to a SKILL.md file.
        """
        return Path(path).name == "SKILL.md"

    def _add_skill_event(self, action: str, path: str | Path):
        """
        Add a skill filesystem event to the async queue.
        """

        path = str(Path(path).expanduser().resolve())

        if not self._is_skill_file(path):
            return

        event_key = (action, path)
        now = time.monotonic()

        last_event = self._last_events.get(event_key)

        if (
            last_event is not None
            and now - last_event < self._debounce_seconds
        ):
            return

        self._last_events[event_key] = now

        if action == "upsert":
            self.known_skill_files.add(path)

        elif action == "delete":
            self.known_skill_files.discard(path)

        event = {
            "action": action,
            "path": path,
        }

        # watchdog runs callbacks from another thread.
        # asyncio.Queue is not thread-safe, so hand the operation
        # back to the asyncio event loop.
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait,
            event,
        )

    def on_created(self, event: FileSystemEvent):
        """
        Handle creation of a new SKILL.md.
        """

        if event.is_directory:
            return

        self._add_skill_event(
            action="upsert",
            path=event.src_path,
        )

    def on_modified(self, event: FileSystemEvent):
        """
        Handle modification of an existing SKILL.md.
        """

        if event.is_directory:
            return

        self._add_skill_event(
            action="upsert",
            path=event.src_path,
        )

    def on_deleted(self, event: FileSystemEvent):
        """
        Handle deletion of a SKILL.md or an entire skill directory.
        """

        if not event.is_directory:
            self._add_skill_event(
                action="delete",
                path=event.src_path,
            )
            return

        deleted_directory = Path(event.src_path).expanduser().resolve()

        # If an entire skill directory disappeared, find any known
        # SKILL.md files that used to live underneath it.
        for skill_path in list(self.known_skill_files):
            path = Path(skill_path)

            try:
                path.relative_to(deleted_directory)
            except ValueError:
                continue

            self._add_skill_event(
                action="delete",
                path=path,
            )

    def on_moved(self, event: FileSystemEvent):
        """
        Handle files/directories being renamed or atomically replaced.
        """

        if not event.is_directory:

            # Old SKILL.md path disappeared.
            if self._is_skill_file(event.src_path):
                self._add_skill_event(
                    action="delete",
                    path=event.src_path,
                )

            # New SKILL.md path appeared.
            if self._is_skill_file(event.dest_path):
                self._add_skill_event(
                    action="upsert",
                    path=event.dest_path,
                )

            return

        source_directory = Path(event.src_path).expanduser().resolve()
        destination_directory = Path(event.dest_path).expanduser().resolve()

        # Handle an entire skill directory being renamed/moved.
        matching_skills = []

        for skill_path in list(self.known_skill_files):
            path = Path(skill_path)

            try:
                relative_path = path.relative_to(source_directory)
            except ValueError:
                continue

            matching_skills.append(
                (
                    path,
                    destination_directory / relative_path,
                )
            )

        for old_path, new_path in matching_skills:

            self._add_skill_event(
                action="delete",
                path=old_path,
            )

            self._add_skill_event(
                action="upsert",
                path=new_path,
            )


class SkillWatcher(DirectoryWatcher):
    """
    Watch a user skills directory for SKILL.md changes.
    """

    def __init__(
        self,
        skills_directory: str,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ):
        self.skills_directory = Path(
            skills_directory
        ).expanduser().resolve()

        super().__init__(
            handler=SkillHandler,
            queue=queue,
            skills_directory=str(self.skills_directory),
            loop=loop,
        )

    def start(self):
        """
        Start recursively watching the skills directory.
        """

        self.skills_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.observer.schedule(
            self.handler,
            path=str(self.skills_directory),
            recursive=True,
        )

        self.observer.start()
