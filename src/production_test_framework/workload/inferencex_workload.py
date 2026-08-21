# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""
InferenceX benchmark workload: runs ``benchmark_serving.py`` in a container via ``docker run``
and parses the "Serving Benchmark Result" block it prints.

The workload is a pure client -- it drives load against an already-running deployment and never
launches a server.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from production_test_framework.ssh import CommandResult
from production_test_framework.workload.command_workload import CommandWorkload, WorkloadCancelled
from production_test_framework.workload.docker_mixin import DockerContainerMixin

__all__ = [
    "DEFAULT_BENCHMARK_OPTIONS",
    "BenchmarkCancelled",
    "InferencexBenchmarkResult",
    "InferencexWorkload",
    "WorkloadCancelled",
    "benchmark_option_argv",
    "parse_benchmark_serving_output",
]

BenchmarkCancelled = WorkloadCancelled

# Applied under whatever the caller passes, so a bare InferencexWorkload() still targets the
# usual single-node vLLM stack. This is data, not logic -- the class attaches no meaning to
# these keys, and a caller can override any of them or drop one entirely by passing None.
#
# To target a disaggregated deployment behind one frontend, pass base_url and drop host/port::
#
#     benchmark_options={"base_url": "http://frontend:8000", "host": None, "port": None}
#
# Dropping them is optional -- benchmark_serving.py prefers base_url when both are given
# (benchmark_serving.py:836-841) -- but it keeps the emitted command honest about the target.
DEFAULT_BENCHMARK_OPTIONS: Mapping[str, Any] = {
    "host": "localhost",
    "port": 8080,
    "model": "Qwen/Qwen3-8B",
    "backend": "vllm",
    "dataset_name": "random",
}


@dataclass
class InferencexBenchmarkResult:
    """
    Parsed "Serving Benchmark Result" block from a ``benchmark_serving.py`` run.

    Every ``Label: <number>`` row in the block lands in :attr:`metrics` under a normalised key,
    so rows added upstream are captured without a change here. The properties below name the
    rows we actually assert on; they are conveniences over the same dict.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    raw_output: str = ""

    def _int(self, key: str) -> int | None:
        value = self.metrics.get(key)
        return None if value is None else int(value)

    @property
    def successful_requests(self) -> int | None:
        return self._int("successful_requests")

    @property
    def total_input_tokens(self) -> int | None:
        return self._int("total_input_tokens")

    @property
    def total_generated_tokens(self) -> int | None:
        return self._int("total_generated_tokens")

    @property
    def duration_seconds(self) -> float | None:
        return self.metrics.get("benchmark_duration_s")

    @property
    def request_throughput(self) -> float | None:
        """Requests per second."""
        return self.metrics.get("request_throughput_req_s")

    @property
    def request_goodput(self) -> float | None:
        """Requests per second meeting the SLOs -- only present when goodput was requested."""
        return self.metrics.get("request_goodput_req_s")

    @property
    def output_token_throughput(self) -> float | None:
        """Generated tokens per second."""
        return self.metrics.get("output_token_throughput_tok_s")

    @property
    def total_token_throughput(self) -> float | None:
        """Input plus generated tokens per second."""
        return self.metrics.get("total_token_throughput_tok_s")

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


_RESULT_BANNER = "Serving Benchmark Result"

# Summary lines are printed as "{:<40} {:<10}" pairs, so a label runs up to the first colon and
# the value is the bare number after it. Lines whose value is not numeric (e.g. "request rate:
# inf") simply do not match and are skipped.
_SUMMARY_LINE_RE = re.compile(r"^(?P<label>\S[^:]*):\s+(?P<value>-?[\d.]+)\s*$")


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _metric_key(label: str) -> str:
    """
    Normalise a summary label into a dict key.

    ``"Successful requests"`` -> ``"successful_requests"``,
    ``"Total Token throughput (tok/s)"`` -> ``"total_token_throughput_tok_s"``,
    ``"P99.9 TTFT (ms)"`` -> ``"p99_9_ttft_ms"``.
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def parse_benchmark_serving_output(output: str) -> InferencexBenchmarkResult:
    """
    Parse the summary block ``benchmark_serving.py`` prints when a run finishes.

    Every ``Label: <number>`` row is captured under a normalised key rather than matched against
    a list of labels we know about, so added rows and extra percentiles are picked up for free.
    Only lines after the "Serving Benchmark Result" banner are considered, so progress output
    above it cannot be mistaken for a summary field. A truncated or absent block yields a result
    with no metrics rather than raising.
    """
    result = InferencexBenchmarkResult(raw_output=output)

    lines = output.splitlines()
    for index, line in enumerate(lines):
        if _RESULT_BANNER in line:
            lines = lines[index + 1 :]
            break
    else:
        return result

    for line in lines:
        match = _SUMMARY_LINE_RE.match(line.strip())
        if match is None:
            continue
        value = _to_float(match.group("value"))
        if value is not None:
            result.metrics[_metric_key(match.group("label"))] = value

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

    The cost of not enumerating flags is that a typo is not caught here. It fails fast and
    legibly instead: argparse rejects it, the container exits non-zero, and ``CommandWorkload``
    surfaces "unrecognized arguments" from stderr as the workload error.
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

    def build_command(self) -> list[str]:
        # --network host is what lets a host/port endpoint reach a server published on the
        # host; everything else about the docker invocation is the base class's.
        return [*self.docker_run_argv("--network", "host"), *self._benchmark_inner_argv()]

    def parse_output(self, result: CommandResult) -> InferencexBenchmarkResult:
        return parse_benchmark_serving_output(result.stdout)

    def _empty_result(self) -> InferencexBenchmarkResult | None:
        return None
