# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Shared base for workloads that run a single cancellable subprocess in a background thread."""

import logging
import threading
import time
from abc import abstractmethod
from concurrent.futures import CancelledError, Future
from typing import Any

from production_test_framework.helper import run_cancellable_command
from production_test_framework.ssh import CommandResult
from production_test_framework.workload.workload import Workload, WorkloadResult, WorkloadStatus


class WorkloadCancelled(Exception):
    """The command run was terminated via ``CommandWorkload.stop()``."""


class CommandWorkload(Workload):
    """
    Base class for workloads whose work is "run one command, capture its output".

    Subclasses only need to supply :meth:`build_command` (the argv to run) and, optionally,
    :meth:`parse_output` (turn the raw ``CommandResult`` into a structured result). Status
    transitions, cancellation, timing, and result plumbing are handled here.
    """

    workload_name = "Command"

    def __init__(
        self,
        *,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
        max_workers: int = 1,
    ):
        super().__init__(max_workers=max_workers)
        # Log under the concrete subclass's module rather than this one.
        self.logger = logging.getLogger(type(self).__module__)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._cancel_event = threading.Event()
        self._completion_fut: Future | None = None
        self._result: Any = self._empty_result()
        self._command_result: CommandResult | None = None

    @abstractmethod
    def build_command(self) -> list[str]:
        """Return the argv (no shell) this workload runs."""

    def parse_output(self, result: CommandResult) -> Any:
        """Convert a successful command result into the workload result. Override to structure it."""
        return result.stdout or "(no stdout)"

    def _empty_result(self) -> Any:
        """Value used for ``result`` before a run and after a stop/cancel."""
        return ""

    @property
    def command_result(self) -> CommandResult | None:
        """Raw result of the last command run, for debugging failed runs."""
        return self._command_result

    def start(self):
        """Start the workload"""

        # We currently only support one run at a time per workload instance.
        if self.status == WorkloadStatus.RUNNING:
            raise RuntimeError(f"{self.workload_name} workload already running")

        self.logger.info("Starting %s workload", self.workload_name.lower())
        self._cancel_event.clear()
        self._start_time = time.time()
        self._workload_status = WorkloadStatus.RUNNING
        self._result = self._empty_result()
        self._command_result = None

        self._completion_fut = self.submit_background(self._run_command_sync)
        self._completion_fut.add_done_callback(self._on_command_done)

    def _run_command_sync(self) -> Any:
        cmd = self.build_command()
        self.logger.info("Running: %s", " ".join(cmd))
        result = run_cancellable_command(
            cmd,
            timeout=self._timeout,
            cancel_event=self._cancel_event,
            poll_interval=self._poll_interval,
        )
        self._command_result = result
        if not result.success:
            if self._cancel_event.is_set():
                raise WorkloadCancelled()
            raise RuntimeError(result.stderr or result.stdout or f"{self.workload_name.lower()} command failed")
        return self.parse_output(result)

    def _on_command_done(self, fut: Future) -> None:
        try:
            result = fut.result()
        except WorkloadCancelled:
            self.logger.info("%s workload stopped", self.workload_name)
            self._workload_status = WorkloadStatus.STOPPED
            self._result = self._empty_result()
            return
        except CancelledError:
            self.logger.info("%s workload cancelled", self.workload_name)
            self._workload_status = WorkloadStatus.STOPPED
            return
        except Exception as e:
            self.logger.exception("%s workload failed", self.workload_name)
            self._workload_status = WorkloadStatus.ERROR
            self._result = str(e)
            return
        finally:
            self._end_time = time.time()

        self._result = result
        self._workload_status = WorkloadStatus.COMPLETED

    def stop(self):
        """Stop the workload, terminating the running command."""
        self.logger.info("Stopping %s workload", self.workload_name.lower())
        self._cancel_event.set()
        if self._completion_fut is not None:
            self._completion_fut.cancel()
        self._workload_status = WorkloadStatus.STOPPED
        self._completion_fut = None

    def get_result(self) -> WorkloadResult:
        return WorkloadResult(
            start_time=self._start_time, end_time=self._end_time, result=self._result, status=self.status
        )
