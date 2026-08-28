# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""
InferenceX benchmark workload: runs ``benchmark_serving.py`` in a container via ``docker run``
and reads the JSON result it writes with ``--save-result``.

The workload is a pure client -- it drives load against an already-running deployment and never
launches a server.
"""

import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from production_test_framework.ssh import CommandResult
from production_test_framework.workload.command_workload import CommandWorkload, WorkloadCancelled
from production_test_framework.workload.docker_mixin import DockerContainerMixin

__all__ = [
    "DEFAULT_BENCHMARK_OPTIONS",
    "JSON_RESULT_INDICATOR",
    "BenchmarkCancelled",
    "BenchmarkResultMissing",
    "InferencexBenchmarkResult",
    "InferencexWorkload",
    "WorkloadCancelled",
    "benchmark_option_argv",
    "parse_benchmark_result_json",
]

BenchmarkCancelled = WorkloadCancelled


class BenchmarkResultMissing(RuntimeError):
    """Raised when a run's stdout carries no usable JSON result."""


# Printed immediately before the JSON so the split point is explicit rather than inferred by
# hunting for the last "{" in a stream that also contains a summary table and any warnings the
# script emitted.
JSON_RESULT_INDICATOR = "---INFERENCEX-RESULT-JSON---"

# Applied under whatever the caller passes, so a bare InferencexWorkload() still targets the
# usual single-node vLLM stack. This is data, not logic -- the class attaches no meaning to
# these keys, and a caller can override any of them or drop one entirely by passing None.
#
# To target a disaggregated deployment behind one frontend, pass base_url and drop host/port::
#
#     benchmark_options={"base_url": "http://frontend:8000", "host": None, "port": None}
DEFAULT_BENCHMARK_OPTIONS: Mapping[str, Any] = {
    "host": "localhost",
    "port": 8080,
    "model": "Qwen/Qwen3-8B",
    "backend": "vllm",
    "dataset_name": "random",
    "save_result": True,
    "result_dir": ".",
    "result_filename": "inferencex-result.json",
}


@dataclass
class InferencexBenchmarkResult:
    """
    Parsed JSON result of a ``benchmark_serving.py`` run.

    Every numeric field lands in :attr:`metrics` under a normalised key, so a field added
    upstream is captured without a change here. The properties below name the ones we actually
    assert on and are conveniences over the same dict; several of them differ from the JSON's own
    spelling, which is why they exist at all.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    raw_output: str = ""

    def _int(self, key: str) -> int | None:
        value = self.metrics.get(key)
        return None if value is None else int(value)

    @property
    def successful_requests(self) -> int | None:
        return self._int("completed")

    @property
    def total_input_tokens(self) -> int | None:
        return self._int("total_input_tokens")

    @property
    def total_generated_tokens(self) -> int | None:
        return self._int("total_output_tokens")

    @property
    def duration_seconds(self) -> float | None:
        return self.metrics.get("duration")

    @property
    def request_throughput(self) -> float | None:
        """Requests per second."""
        return self.metrics.get("request_throughput")

    @property
    def request_goodput(self) -> float | None:
        """
        Requests per second meeting the SLOs -- only present when goodput was requested.

        The JSON spells this key ``"request_goodput:"``, trailing colon included; :func:`_metric_key`
        strips it. Do not "fix" that by matching ``request_goodput`` upstream without checking, or
        this silently reads ``None`` forever.
        """
        return self.metrics.get("request_goodput")

    @property
    def output_token_throughput(self) -> float | None:
        """Generated tokens per second."""
        return self.metrics.get("output_throughput")

    @property
    def total_token_throughput(self) -> float | None:
        """Input plus generated tokens per second."""
        return self.metrics.get("total_token_throughput")

    @property
    def latency_ms(self) -> dict[str, float]:
        """Every millisecond metric, keyed without the unit suffix: ``mean_ttft``, ``p99_tpot``."""
        return {key.removesuffix("_ms"): value for key, value in self.metrics.items() if key.endswith("_ms")}

    @property
    def passed(self) -> bool:
        """True when the run reported a duration and completed at least one request."""
        if self.duration_seconds is None:
            return False
        return bool(self.successful_requests)


def _metric_key(label: str) -> str:
    """
    Normalise a JSON field name into a dict key.

    ``"p99.9_ttft_ms"`` -> ``"p99_9_ttft_ms"`` (a fractional percentile keeps its precision
    rather than colliding with ``p99``), and ``"request_goodput:"`` -> ``"request_goodput"``,
    dropping the trailing colon that field carries upstream.
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def parse_benchmark_result_json(output: str) -> InferencexBenchmarkResult:
    """
    Read the JSON result out of a run's stdout.

    The run appends :data:`JSON_RESULT_INDICATOR` and then the file written by ``--save-result``, so
    everything after the last indicator is the result object.

    Raises :class:`BenchmarkResultMissing` when the indicator is absent, when what follows it is not
    valid JSON, or when it is not an object.

    """
    _, indicator, payload = output.rpartition(JSON_RESULT_INDICATOR)
    if not indicator:
        raise BenchmarkResultMissing(
            f"no {JSON_RESULT_INDICATOR} in the benchmark output, so it wrote no JSON result. "
            "Was save_result turned off, or did the run fail before writing it? "
            "The full output is on the workload's command_result."
        )

    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise BenchmarkResultMissing(
            f"the text after {JSON_RESULT_INDICATOR} is not valid JSON: {exc}. "
            "The full output is on the workload's command_result."
        ) from exc

    if not isinstance(data, dict):
        raise BenchmarkResultMissing(
            f"expected a JSON object after {JSON_RESULT_INDICATOR}, got {type(data).__name__}."
        )

    result = InferencexBenchmarkResult(raw_output=output)
    for name, value in data.items():
        # bool is an int subclass, and a flag recorded in the result is not a measurement.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result.metrics[_metric_key(name)] = float(value)
    return result


def _flag_for(key: str) -> str:
    """``"num_prompts"`` -> ``"--num-prompts"``; a key that is already a flag is left alone."""
    if key.startswith("-"):
        return key
    return "--" + key.replace("_", "-")


def benchmark_option_argv(options: Mapping[str, Any]) -> list[str]:
    """
    Convert a mapping of options into command-line arguments.

    The conversion follows argparse's conventions rather than a table of the flags
    ``benchmark_serving.py`` happens to support today, so a flag added upstream works here with
    no change to this file:

    * ``None`` and ``False`` emit nothing -- the upstream default applies, and ``None`` is how a
      caller drops one of :data:`DEFAULT_BENCHMARK_OPTIONS`
    * ``True`` emits a bare flag, for ``store_true`` options
    * a mapping emits ``--flag K=V K=V`` (``--metadata``)
    * a list/tuple/set emits ``--flag V V`` (``--goodput``, ``--lora-modules``)
    * anything else emits ``--flag str(value)``

    Note that ``0`` and ``""`` are emitted: only ``None`` and ``False`` are skipped, so
    ``{"seed": 0}`` still produces ``--seed 0``.
    """
    args: list[str] = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = _flag_for(key)
        if value is True:
            args.append(flag)
        elif isinstance(value, Mapping):
            args.append(flag)
            args.extend(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, (list, tuple, set)):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.extend([flag, str(value)])
    return args


class InferencexWorkload(DockerContainerMixin, CommandWorkload):
    """
    Run the InferenceX / vLLM ``benchmark_serving.py`` workload inside a container on the
    same host (requires ``docker`` on PATH; often used with a mounted Docker socket).

    The image must provide ``benchmark_script`` at the given path; adjust defaults to match
    your InferenceX container layout.
    """

    workload_name = "Inferencex"
    container_name_prefix = "inferencex"

    def __init__(
        self,
        *,
        image_name: str = "openmosaic/inferencex:latest",
        container_name: str | None = None,
        benchmark_script: str = "/workspace/InferenceX/utils/bench_serving/benchmark_serving.py",
        python_executable: str = "python3",
        benchmark_options: Mapping[str, Any] | None = None,
        benchmark_extra_args: tuple[str, ...] = (),
        docker_exec_timeout: float = 600.0,
        env: Mapping[str, str] | None = None,
        docker_extra_args: tuple[str, ...] = (),
    ):
        super().__init__(
            image_name=image_name,
            container_name=container_name,
            env=env,
            docker_extra_args=docker_extra_args,
            timeout=docker_exec_timeout,
        )

        self._benchmark_script = benchmark_script
        self._python_executable = python_executable
        self._benchmark_options = {**DEFAULT_BENCHMARK_OPTIONS, **(benchmark_options or {})}
        self._benchmark_extra_args = benchmark_extra_args
        self._docker_exec_timeout = docker_exec_timeout

    @property
    def benchmark_options(self) -> dict[str, Any]:
        """The effective options, defaults included -- useful for logging what a run was given."""
        return dict(self._benchmark_options)

    def _benchmark_inner_argv(self) -> list[str]:
        return [
            self._python_executable,
            self._benchmark_script,
            *benchmark_option_argv(self._benchmark_options),
            # Last, so an explicit escape-hatch flag can override anything set above.
            *self._benchmark_extra_args,
        ]

    def _result_path(self) -> str | None:
        """Where ``--save-result`` will write, or None when the caller turned it off."""
        options = self._benchmark_options
        if not options.get("save_result"):
            return None
        filename = options.get("result_filename")
        if not filename:
            return None
        directory = options.get("result_dir")
        return f"{directory}/{filename}" if directory else str(filename)

    def build_command(self) -> list[str]:
        # --network host is what lets a host/port endpoint reach a server published on the
        # host; everything else about the docker invocation is the base class's.
        inner = self._benchmark_inner_argv()
        path = self._result_path()
        if path is not None:
            script = f"{shlex.join(inner)} && echo {shlex.quote(JSON_RESULT_INDICATOR)} && cat {shlex.quote(path)}"
            inner = ["sh", "-c", script]
        return [*self.docker_run_argv("--network", "host"), *inner]

    def parse_output(self, result: CommandResult) -> InferencexBenchmarkResult:
        return parse_benchmark_result_json(result.stdout)

    def _empty_result(self) -> InferencexBenchmarkResult | None:
        return None
