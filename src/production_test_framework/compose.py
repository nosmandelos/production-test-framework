# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Docker Compose helper for compose-based test suites.

ComposeStack brings a docker-compose project's containers up and down and reports their
state through one object, so tests manage the stack instead of shelling out to
`docker compose` by hand.
"""

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ContainerStatus:
    service: str
    name: str
    state: str
    health: str  # "" when the service declares no healthcheck

    @property
    def is_running(self) -> bool:
        return self.state == "running"

    @property
    def is_healthy(self) -> bool:
        if self.health:
            return self.health == "healthy"
        return self.is_running


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


class ComposeStack:
    """A docker compose project - its files and profiles - that can be brought up/down."""

    def __init__(self, files: list[str], *, profiles: tuple[str, ...] = ()) -> None:
        self._files = files
        self._profiles = profiles

    def _args(self, *rest: str) -> list[str]:
        args = ["docker", "compose"]
        for f in self._files:
            args += ["-f", f]
        for profile in self._profiles:
            args += ["--profile", profile]
        return args + list(rest)

    def up(self, *, wait_timeout: int = 180) -> None:
        """Bring the stack up detached and wait for containers to be up/healthy."""
        result = _run(self._args("up", "-d", "--wait", "--wait-timeout", str(wait_timeout)), timeout=wait_timeout + 60)
        if result.returncode != 0:
            raise RuntimeError(f"`docker compose up` failed (rc={result.returncode}):\n{result.stderr}")

    def down(self, *, volumes: bool = True) -> None:
        """Tear the stack down, removing orphans (and volumes by default)."""
        rest = ["down", "--remove-orphans", *(["-v"] if volumes else [])]
        result = _run(self._args(*rest))
        if result.returncode != 0:
            # surface teardown failures: a left-up stack breaks the next run's bring-up
            raise RuntimeError(f"`docker compose down` failed (rc={result.returncode}):\n{result.stderr}")

    def ps(self) -> list[ContainerStatus]:
        """Return the status of every container in the compose project."""
        result = _run(self._args("ps", "--all", "--format", "json"), timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"`docker compose ps` failed (rc={result.returncode}):\n{result.stderr}")
        text = result.stdout.strip()
        if not text:
            return []
        # Newer compose emits one JSON object per line; older emits a single array.
        records = (
            json.loads(text)
            if text.startswith("[")
            else [json.loads(line) for line in text.splitlines() if line.strip()]
        )
        return [
            ContainerStatus(
                service=rec.get("Service", ""),
                name=rec.get("Name", ""),
                state=rec.get("State", ""),
                health=rec.get("Health", ""),
            )
            for rec in records
        ]

    def port(self, service: str, container_port: int) -> str:
        """Host "host:port" that the service's container port is published on.

        Read from the running project so the host port isn't hardcoded.
        """
        result = _run(self._args("port", service, str(container_port)), timeout=30)
        out = result.stdout.strip()
        if result.returncode != 0 or not out:
            raise RuntimeError(f"`docker compose port {service} {container_port}` returned nothing:\n{result.stderr}")
        host, _, port = out.rpartition(":")  # e.g. "0.0.0.0:50051"
        if host in ("", "0.0.0.0", "::", "[::]"):
            host = "localhost"
        return f"{host}:{port}"
