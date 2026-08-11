# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""InferenceX benchmark workload: runs benchmark_serving in a container via ``docker run``."""

from production_test_framework.workload.command_workload import CommandWorkload, WorkloadCancelled

BenchmarkCancelled = WorkloadCancelled


class InferencexWorkload(CommandWorkload):
    """
    Run the InferenceX / vLLM ``benchmark_serving.py`` workload inside a container on the
    same host (requires ``docker`` on PATH; often used with a mounted Docker socket).

    The image must provide ``benchmark_script`` at the given path; adjust defaults to match
    your InferenceX container layout. The vLLM server is reached at ``vllm_host`` and
    ``vllm_port`` on the compose/stack network (e.g. service name or ``localhost``).
    """

    workload_name = "Inferencex"

    def __init__(
        self,
        *,
        image_name: str = "openmosaic/inferencex:latest",
        container_name: str = "inferencex",
        vllm_host: str = "localhost",
        vllm_port: int = 8080,
        benchmark_script: str = "/workspace/InferenceX/utils/bench_serving/benchmark_serving.py",
        python_executable: str = "python3",
        model: str = "Qwen/Qwen3-8B",
        backend: str = "vllm",
        dataset_name: str = "random",
        benchmark_extra_args: tuple[str, ...] = (),
        docker_exec_timeout: float = 600.0,
    ):
        super().__init__(timeout=docker_exec_timeout)

        self._container_name = container_name
        self._image_name = image_name
        self._vllm_host = vllm_host
        self._vllm_port = vllm_port
        self._benchmark_script = benchmark_script
        self._python_executable = python_executable
        self._model = model
        self._backend = backend
        self._dataset_name = dataset_name
        self._benchmark_extra_args = benchmark_extra_args
        self._docker_exec_timeout = docker_exec_timeout

    def _benchmark_inner_argv(self) -> list[str]:
        inner: list[str] = [
            self._python_executable,
            self._benchmark_script,
        ]
        inner.extend(
            [
                "--host",
                self._vllm_host,
                "--port",
                str(self._vllm_port),
                "--model",
                self._model,
                "--backend",
                self._backend,
                "--dataset-name",
                self._dataset_name,
            ]
        )
        inner.extend(self._benchmark_extra_args)
        return inner

    def _docker_exec_cmd(self) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "-t",
            "--network",
            "host",
            "--name",
            self._container_name,
            self._image_name,
            *self._benchmark_inner_argv(),
        ]

    def build_command(self) -> list[str]:
        return self._docker_exec_cmd()
