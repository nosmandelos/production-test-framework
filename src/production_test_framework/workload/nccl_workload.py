# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""
NCCL test workload: runs an `nccl-tests <https://github.com/NVIDIA/nccl-tests>`_ performance
binary (``all_reduce_perf``, ``all_gather_perf``, ...) and parses the reported bandwidth table.

The binaries are driven through their documented command-line interface and their plain-text
output is parsed, so no vendor SDK is required. Three launch shapes are supported:

* single node, binaries already on the host or in the current container (the default)
* single node, containerised (``use_docker=True`` plus an ``image_name``), mirroring
  :class:`~production_test_framework.workload.inferencex_workload.InferencexWorkload`
* one or more hosts via ``mpirun`` (pass ``hosts=(...)``), optionally inside the launcher container
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from production_test_framework.ssh import CommandResult
from production_test_framework.workload.command_workload import CommandWorkload, WorkloadCancelled
from production_test_framework.workload.docker_mixin import DockerContainerMixin

__all__ = [
    "DEFAULT_OTEL_ENDPOINT",
    "OTEL_ENDPOINT_ENV",
    "NcclTest",
    "NcclSample",
    "NcclTestResult",
    "NcclWorkload",
    "WorkloadCancelled",
    "parse_nccl_output",
]

OTEL_ENDPOINT_ENV = "NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT"

DEFAULT_OTEL_ENDPOINT = "http://172.17.0.1:4318"


class NcclTest(Enum):
    """nccl-tests performance binaries (value is the binary name)."""

    ALL_REDUCE = "all_reduce_perf"
    ALL_GATHER = "all_gather_perf"
    BROADCAST = "broadcast_perf"
    REDUCE = "reduce_perf"
    REDUCE_SCATTER = "reduce_scatter_perf"
    ALLTOALL = "alltoall_perf"
    SCATTER = "scatter_perf"
    GATHER = "gather_perf"
    SENDRECV = "sendrecv_perf"
    HYPERCUBE = "hypercube_perf"


@dataclass
class NcclSample:
    """One row of the nccl-tests bandwidth table (out-of-place or in-place half)."""

    size_bytes: int
    count: int
    dtype: str
    in_place: bool
    time_us: float | None
    algbw_gbps: float | None
    busbw_gbps: float | None
    wrong: int | None


@dataclass
class NcclTestResult:
    """Parsed result of an nccl-tests run."""

    test: str
    samples: list[NcclSample] = field(default_factory=list)
    avg_bus_bandwidth_gbps: float | None = None
    out_of_bounds_errors: int | None = None
    raw_output: str = ""

    @property
    def max_busbw_gbps(self) -> float | None:
        """Highest bus bandwidth observed across all message sizes."""
        values = [s.busbw_gbps for s in self.samples if s.busbw_gbps is not None]
        return max(values) if values else None

    @property
    def passed(self) -> bool:
        """True when the run produced samples and reported no correctness errors."""
        if not self.samples:
            return False
        if self.out_of_bounds_errors:
            return False
        return all(s.wrong in (0, None) for s in self.samples)


_AVG_BW_RE = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([0-9.]+)")
_OUT_OF_BOUNDS_RE = re.compile(r"#\s*Out of bounds values\s*:\s*(\d+)")


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def parse_nccl_output(output: str, test: str = "") -> NcclTestResult:
    """
    Parse the nccl-tests bandwidth table.

    Data rows are ``size count type [redop [root]] <time algbw busbw #wrong> x2`` -- the middle
    columns vary per test (``all_gather_perf`` has no redop/root), so the two result groups are
    read from the end of the row and the size/count/type from the front.
    """
    result = NcclTestResult(test=test, raw_output=output)

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            if (avg := _AVG_BW_RE.search(stripped)) is not None:
                result.avg_bus_bandwidth_gbps = _to_float(avg.group(1))
            elif (oob := _OUT_OF_BOUNDS_RE.search(stripped)) is not None:
                result.out_of_bounds_errors = _to_int(oob.group(1))
            continue

        fields = stripped.split()
        # 3 leading columns + two groups of (time, algbw, busbw, #wrong)
        if len(fields) < 11:
            continue
        size_bytes = _to_int(fields[0])
        count = _to_int(fields[1])
        if size_bytes is None or count is None:
            continue

        dtype = fields[2]
        for in_place, group in ((False, fields[-8:-4]), (True, fields[-4:])):
            time_us, algbw, busbw, wrong = group
            result.samples.append(
                NcclSample(
                    size_bytes=size_bytes,
                    count=count,
                    dtype=dtype,
                    in_place=in_place,
                    time_us=_to_float(time_us),
                    algbw_gbps=_to_float(algbw),
                    busbw_gbps=_to_float(busbw),
                    wrong=_to_int(wrong),
                )
            )

    return result


class NcclWorkload(DockerContainerMixin, CommandWorkload):
    """
    Run an nccl-tests performance binary and report the bandwidth it measured.

    Runs in ``openmosaic/mosaic-nccl-tests`` by default -- nccl-tests built with ``MPI=1`` under
    ``/workspace/bin`` alongside the Mosaic NCCL OTEL profiler plugin, from
    https://github.com/open-mosaic/nccl-tests-docker. Set ``use_docker=False`` to run binaries
    already present under ``binary_dir``. There is no official upstream nccl-tests image, so
    ``image_name`` is required whenever ``use_docker`` is set.

    Passing ``hosts`` switches the launch to ``mpirun`` with one rank per GPU on every host, per
    the nccl-tests guidance. Without it a single process drives all ``gpus_per_host`` GPUs.

    Example::

        wl = NcclWorkload(test=NcclTest.ALL_REDUCE, gpus_per_host=4, max_bytes="128M")
        wl.start()
        wl.wait_for_completion(timeout=900)
        print(wl.get_result().result.avg_bus_bandwidth_gbps)
    """

    workload_name = "NCCL"
    container_name_prefix = "nccl-tests"

    def __init__(
        self,
        *,
        test: NcclTest = NcclTest.ALL_REDUCE,
        image_name: str | None = "openmosaic/mosaic-nccl-tests:latest",
        container_name: str | None = None,
        binary_dir: str = "/workspace/bin",
        gpus_per_host: int = 8,
        min_bytes: str = "8",
        max_bytes: str = "512M",
        step_factor: int = 2,
        iters: int = 3000,
        warmup_iters: int = 5,
        check: bool = True,
        hosts: tuple[str, ...] = (),
        num_processes: int | None = None,
        env: dict[str, str] | None = None,
        otel_endpoint: str | None = DEFAULT_OTEL_ENDPOINT,
        use_docker: bool = True,
        gpus: str = "all",
        docker_network: str | None = None,
        docker_extra_args: tuple[str, ...] = (),
        mpi_extra_args: tuple[str, ...] = (),
        test_extra_args: tuple[str, ...] = (),
        timeout: float = 600.0,
    ):
        if use_docker and not image_name:
            raise ValueError("use_docker=True requires image_name (there is no default nccl-tests image)")

        # Built before super() so the base owns the finished mapping: the profiler endpoint is
        # part of the run's environment, not something layered on afterwards.
        run_env = dict(env or {})
        if otel_endpoint and OTEL_ENDPOINT_ENV not in run_env:
            run_env[OTEL_ENDPOINT_ENV] = otel_endpoint

        super().__init__(
            image_name=image_name,
            container_name=container_name,
            env=run_env,
            docker_extra_args=docker_extra_args,
            timeout=timeout,
        )

        self._test = test
        self._binary_dir = binary_dir.rstrip("/")
        self._gpus_per_host = gpus_per_host
        self._min_bytes = min_bytes
        self._max_bytes = max_bytes
        self._step_factor = step_factor
        self._iters = iters
        self._warmup_iters = warmup_iters
        self._check = check
        self._hosts = tuple(hosts)
        self._num_processes = num_processes
        self._use_docker = use_docker
        self._gpus = gpus
        self._docker_network = docker_network
        self._mpi_extra_args = mpi_extra_args
        self._test_extra_args = test_extra_args

    @property
    def test(self) -> NcclTest:
        return self._test

    @property
    def uses_docker(self) -> bool:
        """False when the binaries are run directly on the host, so there is nothing to clean up."""
        return self._use_docker

    @property
    def gpus(self) -> str:
        """
        Value for ``docker run --gpus``: ``all`` (default), a count, or a device selection
        such as ``device=2,3`` to pin the run to specific host GPUs.
        """
        if "," in self._gpus and not self._gpus.startswith('"'):
            return f'"{self._gpus}"'
        return self._gpus

    @property
    def docker_network(self) -> str:
        """
        Value for ``docker run --network``, derived from the launch shape unless set explicitly.

        Single-container runs get ``bridge``: the image sets ``NCCL_SOCKET_IFNAME=eth0``, and only
        bridge networking gives the container an ``eth0`` -- under ``host`` NCCL's bootstrap finds
        no matching interface and aborts. An ``mpirun`` launch gets ``host`` so ranks can reach
        their peers.
        """
        if self._docker_network is not None:
            return self._docker_network
        return "host" if self.use_mpi else "bridge"

    @property
    def use_mpi(self) -> bool:
        """True when the run is launched with ``mpirun`` -- i.e. whenever ``hosts`` is set."""
        return bool(self._hosts)

    @property
    def hosts(self) -> tuple[str, ...]:
        return self._hosts

    @property
    def num_processes(self) -> int:
        """
        Processes the run launches.

        Under ``mpirun`` that is one rank per GPU across every host in ``hosts``; a standalone
        run is a single process driving all ``gpus_per_host`` GPUs (see ``-g`` in the argv).
        """
        if self._num_processes is not None:
            return self._num_processes
        if not self.use_mpi:
            return 1
        return len(self._hosts) * self._gpus_per_host

    def _docker_argv(self) -> list[str]:
        # nccl-tests needs the memlock and stack limits raised for large message sizes, and
        # host IPC for shared-memory transports between ranks. Naming, labelling, environment
        # and the image are the base class's.
        return self.docker_run_argv(
            "--gpus",
            self.gpus,
            "--network",
            self.docker_network,
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
        )

    def _mpi_argv(self) -> list[str]:
        host_list = ",".join(f"{host}:{self._gpus_per_host}" for host in self._hosts)
        return [
            "mpirun",
            "--allow-run-as-root",
            "-np",
            str(self.num_processes),
            "-H",
            host_list,
            *self.env_args("-x"),
            *self._mpi_extra_args,
        ]

    def _test_argv(self) -> list[str]:
        # Under mpirun each rank owns a single GPU; standalone runs drive all GPUs from one process.
        gpus_per_process = 1 if self.use_mpi else self._gpus_per_host
        return [
            f"{self._binary_dir}/{self._test.value}",
            "-b",
            self._min_bytes,
            "-e",
            self._max_bytes,
            "-f",
            str(self._step_factor),
            "-g",
            str(gpus_per_process),
            "-n",
            str(self._iters),
            "-w",
            str(self._warmup_iters),
            "-c",
            "1" if self._check else "0",
            *self._test_extra_args,
        ]

    def build_command(self) -> list[str]:
        cmd: list[str] = []
        if self._use_docker:
            cmd.extend(self._docker_argv())
        if self.use_mpi:
            cmd.extend(self._mpi_argv())
        cmd.extend(self._test_argv())
        return cmd

    def parse_output(self, result: CommandResult) -> NcclTestResult:
        # NCCL_DEBUG output goes to stderr; the bandwidth table can land on either stream
        # depending on how the run was launched, so parse whichever carries it.
        parsed = parse_nccl_output(result.stdout, test=self._test.value)
        if not parsed.samples and result.stderr:
            parsed = parse_nccl_output(result.stderr, test=self._test.value)
        return parsed

    def _empty_result(self) -> NcclTestResult | None:
        return None
