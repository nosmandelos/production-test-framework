# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

import logging
import time
from concurrent.futures import CancelledError, Future
from enum import Enum

from production_test_framework.vllm import DEFAULT_MODEL, VllmClient, VllmConfig
from production_test_framework.workload.workload import Workload, WorkloadResult, WorkloadStatus


class BACKEND_TYPE(Enum):
    VLLM = "vllm"


class PromptWorkload(Workload):
    def __init__(
        self,
        prompt: str,
        backend_type: BACKEND_TYPE = BACKEND_TYPE.VLLM,
        host: str = "localhost",
        port: int = 8080,
        model: str = DEFAULT_MODEL,
    ):
        """
        Send one prompt to a backend and hold the result.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.prompt = prompt
        self.backend = None
        self._completion_fut: Future | None = None
        self.prompt_state = WorkloadStatus.STOPPED
        self.prompt_result = None

        match backend_type:
            case BACKEND_TYPE.VLLM:
                self.backend = VllmClient(VllmConfig(host=host, port=port, model=model))

    @property
    def status(self) -> WorkloadStatus:
        return self.prompt_state

    def start(self):
        """Start the prompt workload"""

        # We currently only support one prompt workload at a time.
        if self.status == WorkloadStatus.RUNNING:
            raise RuntimeError("Prompt workload already running")

        self.logger.info("Starting prompt workload: %s", self.prompt)
        self.logger.info("waiting for backend to be ready...")
        self.backend.wait_for_ready(timeout=30)

        self.logger.info("sending prompt to backend...")
        self._start_time = time.time()
        self.prompt_state = WorkloadStatus.RUNNING
        self.prompt_result = None
        self._completion_fut = self.submit_background(self.backend.complete, self.prompt)
        self._completion_fut.add_done_callback(self._on_completion_done)

    def stop(self):
        """Stop the prompt workload"""
        self.logger.info("Stopping prompt workload...")
        self._completion_fut.cancel()
        self.prompt_state = WorkloadStatus.STOPPED
        self.prompt_result = None
        self._completion_fut = None
        self.logger.info("Prompt workload stopped")

    def get_result(self) -> str:
        """Return inference text after the prompt task has completed."""
        return WorkloadResult(
            start_time=self._start_time, end_time=self._end_time, result=self.prompt_result, status=self.status
        )

    def _on_completion_done(self, fut: Future) -> None:
        try:
            result = fut.result()
        except CancelledError:
            self.logger.info("Prompt completion cancelled")
            return
        except Exception:
            self.logger.exception("Prompt workload failed")
            self.prompt_state = WorkloadStatus.ERROR
            return
        finally:
            self._end_time = time.time()
            self.prompt_result = result
            self.prompt_state = WorkloadStatus.COMPLETED
        self.logger.info("Prompt workload completed: %s", self.prompt_result)
