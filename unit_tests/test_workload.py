# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for workload base and concrete workload classes."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from production_test_framework.ssh import CommandResult
from production_test_framework.vllm import InferenceResult
from production_test_framework.workload.inferencex_workload import InferencexWorkload
from production_test_framework.workload.nccl_workload import (
    NcclTest,
    NcclWorkload,
    parse_nccl_output,
)
from production_test_framework.workload.prompt_workload import BACKEND_TYPE, PromptWorkload
from production_test_framework.workload.workload import Workload, WorkloadStatus

NCCL_ALL_REDUCE_OUTPUT = """\
# nThread 1 nGpus 8 minBytes 8 maxBytes 268435456 step: 2(factor) warmup iters: 5 iters: 20 validation: 1
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    23.45    0.00    0.00      0    22.11    0.00    0.00      0
   268435456      67108864     float     sum      -1  1234.50  217.45  380.54      0  1230.10  218.22  381.89      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 190.715
#
"""

NCCL_ALL_GATHER_OUTPUT = """\
#       size         count      type     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
   134217728       2097152     float   500.10  268.38  234.83    N/A   498.00  269.51  235.82    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 235.325
"""


class TestWorkloadStatus:
    """Tests for WorkloadStatus enum."""

    def test_member_values(self):
        assert WorkloadStatus.RUNNING.value == "running"
        assert WorkloadStatus.STOPPED.value == "stopped"
        assert WorkloadStatus.COMPLETED.value == "completed"
        assert WorkloadStatus.ERROR.value == "error"


class TestWorkload:
    """Tests for abstract Workload base class."""

    def test_cannot_instantiate_abstract_workload(self):
        with pytest.raises(TypeError, match="abstract"):
            Workload()

    def test_wait_for_completion_returns_true_when_predicate_succeeds(self):
        class CompletingWorkload(Workload):
            def __init__(self):
                super().__init__()

            def start(self):
                self._workload_status = WorkloadStatus.RUNNING

                def finish():
                    self._workload_status = WorkloadStatus.COMPLETED

                self.submit_background(finish)

            def stop(self):
                self._workload_status = WorkloadStatus.STOPPED

            def get_result(self) -> str:
                return "done"

        with patch(
            "production_test_framework.workload.workload.wait_for",
            return_value=True,
        ) as mock_wait:
            wl = CompletingWorkload()
            assert wl.wait_for_completion(timeout=10.0, poll_interval=1.0) is True

        mock_wait.assert_called_once()
        args, kwargs = mock_wait.call_args
        assert kwargs == {}
        pred, timeout_arg, poll_arg = args
        assert timeout_arg == 10.0
        assert poll_arg == 1.0
        wl._workload_status = WorkloadStatus.COMPLETED
        assert pred() is True

    def test_wait_for_completion_returns_false_when_wait_times_out(self):
        class NeverCompletingWorkload(Workload):
            def __init__(self):
                super().__init__()
                self._workload_status = WorkloadStatus.RUNNING

            def start(self):
                pass

            def stop(self):
                pass

            def get_result(self) -> str:
                return ""

        with patch(
            "production_test_framework.workload.workload.wait_for",
            return_value=False,
        ) as mock_wait:
            wl = NeverCompletingWorkload()
            assert wl.wait_for_completion(timeout=5.0, poll_interval=1.0) is False

        mock_wait.assert_called_once()

    def test_status_property_reads_workload_status(self):
        class DummyWorkload(Workload):
            def start(self):
                pass

            def stop(self):
                pass

            def get_result(self) -> str:
                return ""

        w = DummyWorkload()
        assert w.status == WorkloadStatus.STOPPED
        w._workload_status = WorkloadStatus.RUNNING
        assert w.status == WorkloadStatus.RUNNING

    def test_submit_background_runs_callable(self):
        class RunnableWorkload(Workload):
            def __init__(self):
                super().__init__()
                self.seen = []

            def start(self):
                pass

            def stop(self):
                pass

            def get_result(self) -> str:
                return ""

            def capture(self, x):
                self.seen.append(x)

        w = RunnableWorkload()
        try:
            fut = w.submit_background(w.capture, 42)
            assert fut.result(timeout=5.0) is None
            assert w.seen == [42]
        finally:
            w.shutdown_executor(wait=True)


class TestInferencexWorkload:
    """Tests for Inferencex workload."""

    @pytest.fixture
    def mock_inferencex_run(self):
        with patch(
            "production_test_framework.workload.command_workload.run_cancellable_command",
        ) as m:
            m.return_value = CommandResult(
                returncode=0,
                stdout="benchmark output\n",
                stderr="",
            )
            yield m

    def test_is_workload_subclass(self):
        assert issubclass(InferencexWorkload, Workload)

    def test_can_instantiate(self):
        w = InferencexWorkload()
        assert isinstance(w, Workload)

    def test_initial_status_is_stopped(self):
        w = InferencexWorkload()
        assert w.status == WorkloadStatus.STOPPED

    def test_start_transitions_to_running(self, mock_inferencex_run):
        w = InferencexWorkload()
        w.start()
        assert w.status in (WorkloadStatus.RUNNING, WorkloadStatus.COMPLETED)
        w.shutdown_executor(wait=True, cancel_futures=True)

    def test_stop_returns_to_stopped(self, mock_inferencex_run):
        w = InferencexWorkload()
        w.start()
        w.stop()
        assert w.status == WorkloadStatus.STOPPED
        _args, kwargs = mock_inferencex_run.call_args
        assert kwargs["cancel_event"].is_set()
        w.shutdown_executor(wait=True)

    def test_get_result_after_completion(self, mock_inferencex_run):
        w = InferencexWorkload()
        w.start()
        fut = w._completion_fut
        assert fut is not None
        fut.result(timeout=10.0)
        assert w.status == WorkloadStatus.COMPLETED
        assert w.get_result().result == "benchmark output\n"
        assert w.get_result().status == WorkloadStatus.COMPLETED
        assert w.get_result().start_time is not None
        assert w.get_result().end_time is not None
        assert w.get_result().runtime is not None
        w.shutdown_executor(wait=True)

    def test_docker_exec_argv_includes_container_host_port(self, mock_inferencex_run):
        w = InferencexWorkload(
            container_name="mycontainer",
            vllm_host="vllm.svc",
            vllm_port=9090,
        )
        w.start()
        w._completion_fut.result(timeout=10.0)
        cmd = mock_inferencex_run.call_args[0][0]
        assert cmd[:2] == ["docker", "run"]
        assert "--host" in cmd
        assert "vllm.svc" in cmd
        assert "--port" in cmd
        assert "9090" in cmd
        assert "--base-url" not in cmd
        w.shutdown_executor(wait=True)

    def test_stop_while_running_sets_cancel_on_mock(self, mock_inferencex_run):
        def run_until_cancel(cmd, *, timeout, cancel_event, **kwargs):
            for _ in range(500):
                if cancel_event.is_set():
                    return CommandResult(returncode=-1, stdout="", stderr="cancelled")
                time.sleep(0.01)
            return CommandResult(returncode=0, stdout="done", stderr="")

        mock_inferencex_run.side_effect = run_until_cancel
        w = InferencexWorkload()
        w.start()
        time.sleep(0.05)
        w.stop()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.status == WorkloadStatus.STOPPED and w.get_result() == "":
                break
            time.sleep(0.02)
        assert w.status == WorkloadStatus.STOPPED
        assert w.get_result().result == ""
        assert w.get_result().status == WorkloadStatus.STOPPED
        assert w.get_result().start_time is not None
        assert w.get_result().end_time is not None
        assert w.get_result().runtime is not None
        w.shutdown_executor(wait=True)

    def test_second_start_raises_when_already_running(self, mock_inferencex_run):
        block = threading.Event()

        def slow_run(*_args, **_kwargs):
            block.wait(timeout=60.0)
            return CommandResult(returncode=0, stdout="done", stderr="")

        mock_inferencex_run.side_effect = slow_run
        w = InferencexWorkload()
        w.start()
        assert w.status == WorkloadStatus.RUNNING
        with pytest.raises(
            RuntimeError,
            match="Inferencex workload already running",
        ):
            w.start()
        block.set()
        w._completion_fut.result(timeout=10.0)
        w.shutdown_executor(wait=True)


class TestNcclOutputParsing:
    """Tests for the nccl-tests output parser."""

    def test_parses_samples_and_summary(self):
        result = parse_nccl_output(NCCL_ALL_REDUCE_OUTPUT, test="all_reduce_perf")
        assert result.test == "all_reduce_perf"
        assert result.avg_bus_bandwidth_gbps == 190.715
        assert result.out_of_bounds_errors == 0
        # two data rows x (out-of-place, in-place)
        assert len(result.samples) == 4
        largest = [s for s in result.samples if s.size_bytes == 268435456]
        assert {s.in_place for s in largest} == {True, False}
        out_of_place = next(s for s in largest if not s.in_place)
        assert out_of_place.count == 67108864
        assert out_of_place.dtype == "float"
        assert out_of_place.time_us == 1234.50
        assert out_of_place.algbw_gbps == 217.45
        assert out_of_place.busbw_gbps == 380.54
        assert out_of_place.wrong == 0

    def test_max_busbw_and_passed(self):
        result = parse_nccl_output(NCCL_ALL_REDUCE_OUTPUT)
        assert result.max_busbw_gbps == 381.89
        assert result.passed is True

    def test_parses_rows_without_redop_and_root_columns(self):
        result = parse_nccl_output(NCCL_ALL_GATHER_OUTPUT, test="all_gather_perf")
        assert len(result.samples) == 2
        assert result.avg_bus_bandwidth_gbps == 235.325
        assert result.max_busbw_gbps == 235.82
        # validation disabled -> "N/A" is not an error
        assert all(s.wrong is None for s in result.samples)
        assert result.passed is True

    def test_out_of_bounds_errors_fail_the_run(self):
        output = NCCL_ALL_REDUCE_OUTPUT.replace("# Out of bounds values : 0 OK", "# Out of bounds values : 2 FAILED")
        result = parse_nccl_output(output)
        assert result.out_of_bounds_errors == 2
        assert result.passed is False

    def test_empty_output_has_no_samples_and_does_not_pass(self):
        result = parse_nccl_output("")
        assert result.samples == []
        assert result.avg_bus_bandwidth_gbps is None
        assert result.passed is False


class TestNcclWorkload:
    """Tests for NCCL test workload."""

    @pytest.fixture
    def mock_nccl_run(self):
        with patch(
            "production_test_framework.workload.command_workload.run_cancellable_command",
        ) as m:
            m.return_value = CommandResult(
                returncode=0,
                stdout=NCCL_ALL_REDUCE_OUTPUT,
                stderr="",
            )
            yield m

    def test_is_workload_subclass(self):
        assert issubclass(NcclWorkload, Workload)

    def test_initial_status_is_stopped(self):
        w = NcclWorkload()
        assert w.status == WorkloadStatus.STOPPED
        assert w.get_result().result is None

    def test_default_command_runs_binary_in_container(self):
        w = NcclWorkload()
        cmd = w.build_command()
        assert cmd[:2] == ["docker", "run"]
        assert "--gpus" in cmd and "all" in cmd
        assert "openmosaic/mosaic-nccl-tests:latest" in cmd
        assert "/workspace/bin/all_reduce_perf" in cmd
        assert "mpirun" not in cmd
        # single-node run drives every GPU from one process
        assert cmd[cmd.index("-g") + 1] == "8"

    def test_containerised_run_requires_an_image(self):
        with pytest.raises(ValueError, match="requires image_name"):
            NcclWorkload(use_docker=True, image_name=None)

        w = NcclWorkload(image_name="registry.local/nccl-tests:v2.13")
        assert "registry.local/nccl-tests:v2.13" in w.build_command()

    def test_runs_binary_directly_without_docker(self):
        w = NcclWorkload(use_docker=False, image_name=None)
        cmd = w.build_command()
        assert cmd[0] == "/workspace/bin/all_reduce_perf"
        assert "docker" not in cmd

    def test_gpus_defaults_to_all(self):
        w = NcclWorkload()
        cmd = w.build_command()
        assert cmd[cmd.index("--gpus") + 1] == "all"

    def test_gpu_selection_is_quoted_for_dockers_csv_parser(self):
        # Bare "device=2,3" is split on the comma and rejected by dockerd with
        # "cannot set both Count and DeviceIDs on device request".
        w = NcclWorkload(gpus="device=2,3", gpus_per_host=2)
        assert w.gpus == '"device=2,3"'
        cmd = w.build_command()
        assert cmd[cmd.index("--gpus") + 1] == '"device=2,3"'
        # GPUs are renumbered from 0 in the container, so -g counts them
        assert cmd[cmd.index("-g") + 1] == "2"

    def test_single_gpu_selection_needs_no_quoting(self):
        assert NcclWorkload(gpus="device=2").gpus == "device=2"
        assert NcclWorkload(gpus="2").gpus == "2"

    def test_already_quoted_gpu_selection_is_left_alone(self):
        assert NcclWorkload(gpus='"device=0,1"').gpus == '"device=0,1"'

    def test_network_defaults_to_bridge_so_the_image_finds_eth0(self):
        # The image sets NCCL_SOCKET_IFNAME=eth0; --network host has no eth0 and NCCL's
        # bootstrap aborts with "no socket interface found".
        w = NcclWorkload()
        assert w.docker_network == "bridge"
        cmd = w.build_command()
        assert cmd[cmd.index("--network") + 1] == "bridge"

    def test_network_defaults_to_host_under_mpirun(self):
        w = NcclWorkload(hosts=("gpu01", "gpu02"))
        assert w.docker_network == "host"
        cmd = w.build_command()
        assert cmd[cmd.index("--network") + 1] == "host"

    def test_network_can_be_overridden(self):
        w = NcclWorkload(docker_network="my-net")
        assert w.docker_network == "my-net"
        assert w.build_command()[w.build_command().index("--network") + 1] == "my-net"

    def test_test_selection_and_sizes(self):
        w = NcclWorkload(
            test=NcclTest.ALL_GATHER,
            binary_dir="/usr/local/nccl-tests/build/",
            min_bytes="1M",
            max_bytes="4G",
            step_factor=4,
            iters=50,
            warmup_iters=10,
            check=False,
            test_extra_args=("-z", "1"),
        )
        # Read the flags from the binary onwards: `docker run` takes its own -e (env), which
        # would otherwise shadow the test binary's -e (max bytes).
        full = w.build_command()
        assert "/usr/local/nccl-tests/build/all_gather_perf" in full
        cmd = full[full.index("/usr/local/nccl-tests/build/all_gather_perf") :]
        assert cmd[cmd.index("-b") + 1] == "1M"
        assert cmd[cmd.index("-e") + 1] == "4G"
        assert cmd[cmd.index("-f") + 1] == "4"
        assert cmd[cmd.index("-n") + 1] == "50"
        assert cmd[cmd.index("-w") + 1] == "10"
        assert cmd[cmd.index("-c") + 1] == "0"
        assert cmd[-2:] == ["-z", "1"]

    def test_hosts_switch_the_run_to_mpirun_with_one_gpu_per_rank(self):
        w = NcclWorkload(
            hosts=("gpu01", "gpu02"),
            gpus_per_host=4,
            env={"NCCL_DEBUG": "INFO"},
            mpi_extra_args=("--mca", "btl", "tcp,self"),
        )
        assert w.use_mpi is True
        assert w.num_processes == 8
        cmd = w.build_command()
        # mpirun is the launcher inside the container, so it follows the image name
        assert cmd[cmd.index("openmosaic/mosaic-nccl-tests:latest") + 1] == "mpirun"
        assert cmd[cmd.index("-np") + 1] == "8"
        assert cmd[cmd.index("-H") + 1] == "gpu01:4,gpu02:4"
        assert cmd[cmd.index("-x") + 1] == "NCCL_DEBUG=INFO"
        assert "--mca" in cmd
        assert cmd[cmd.index("-g") + 1] == "1"

    def test_single_host_still_launches_under_mpirun(self):
        w = NcclWorkload(hosts=("gpu01",), gpus_per_host=8, use_docker=False, image_name=None)
        assert w.use_mpi is True
        cmd = w.build_command()
        assert cmd[0] == "mpirun"
        assert cmd[cmd.index("-np") + 1] == "8"
        assert cmd[cmd.index("-H") + 1] == "gpu01:8"
        assert cmd[cmd.index("-g") + 1] == "1"

    def test_no_hosts_means_no_mpirun(self):
        w = NcclWorkload(gpus_per_host=8)
        assert w.use_mpi is False
        assert w.hosts == ()
        assert "mpirun" not in w.build_command()
        # one process drives all 8 GPUs, so the process count is 1 -- not 0, not 8
        assert w.num_processes == 1

    def test_num_processes_override(self):
        w = NcclWorkload(hosts=("gpu01",), gpus_per_host=8, num_processes=2)
        assert w.num_processes == 2
        assert w.build_command()[w.build_command().index("-np") + 1] == "2"

    def test_env_is_passed_to_container_and_mpirun(self):
        w = NcclWorkload(
            hosts=("gpu01",),
            env={"NCCL_IB_DISABLE": "0"},
            use_docker=True,
            image_name="registry.local/nccl-tests:v2.13",
        )
        cmd = w.build_command()
        assert cmd[cmd.index("-e") + 1] == "NCCL_IB_DISABLE=0"
        assert cmd[cmd.index("-x") + 1] == "NCCL_IB_DISABLE=0"

    def test_get_result_after_completion_is_parsed(self, mock_nccl_run):
        w = NcclWorkload()
        w.start()
        w._completion_fut.result(timeout=10.0)
        assert w.status == WorkloadStatus.COMPLETED
        result = w.get_result()
        assert result.status == WorkloadStatus.COMPLETED
        assert result.result.test == "all_reduce_perf"
        assert result.result.avg_bus_bandwidth_gbps == 190.715
        assert result.result.passed is True
        assert result.runtime is not None
        w.shutdown_executor(wait=True)

    def test_parses_table_from_stderr_when_stdout_empty(self, mock_nccl_run):
        mock_nccl_run.return_value = CommandResult(
            returncode=0,
            stdout="",
            stderr=NCCL_ALL_REDUCE_OUTPUT,
        )
        w = NcclWorkload()
        w.start()
        w._completion_fut.result(timeout=10.0)
        assert w.get_result().result.avg_bus_bandwidth_gbps == 190.715
        w.shutdown_executor(wait=True)

    def test_failed_run_sets_error_status(self, mock_nccl_run):
        mock_nccl_run.return_value = CommandResult(
            returncode=1,
            stdout="",
            stderr="NCCL failure: unhandled system error",
        )
        w = NcclWorkload()
        w.start()
        assert w.wait_for_completion(timeout=10.0, poll_interval=0.05) is True
        assert w.status == WorkloadStatus.ERROR
        assert "unhandled system error" in w.get_result().result
        w.shutdown_executor(wait=True)

    def test_stop_cancels_and_clears_result(self, mock_nccl_run):
        def run_until_cancel(cmd, *, timeout, cancel_event, **kwargs):
            for _ in range(500):
                if cancel_event.is_set():
                    return CommandResult(returncode=-1, stdout="", stderr="cancelled")
                time.sleep(0.01)
            return CommandResult(returncode=0, stdout=NCCL_ALL_REDUCE_OUTPUT, stderr="")

        mock_nccl_run.side_effect = run_until_cancel
        w = NcclWorkload()
        w.start()
        time.sleep(0.05)
        w.stop()
        assert mock_nccl_run.call_args[1]["cancel_event"].is_set()
        assert w.status == WorkloadStatus.STOPPED
        assert w.get_result().result is None
        w.shutdown_executor(wait=True)

    def test_second_start_raises_when_already_running(self, mock_nccl_run):
        block = threading.Event()

        def slow_run(*_args, **_kwargs):
            block.wait(timeout=60.0)
            return CommandResult(returncode=0, stdout=NCCL_ALL_REDUCE_OUTPUT, stderr="")

        mock_nccl_run.side_effect = slow_run
        w = NcclWorkload()
        w.start()
        assert w.status == WorkloadStatus.RUNNING
        with pytest.raises(RuntimeError, match="NCCL workload already running"):
            w.start()
        block.set()
        w._completion_fut.result(timeout=10.0)
        w.shutdown_executor(wait=True)


class TestPromptWorkload:
    """Tests for prompt-driven workload against a backend."""

    @pytest.fixture
    def mock_vllm_client_class(self):
        with patch(
            "production_test_framework.workload.prompt_workload.VllmClient",
        ) as m:
            backend = MagicMock()
            backend.wait_for_ready = MagicMock(return_value=True)
            backend.complete = MagicMock(return_value=InferenceResult(success=True, text="model output"))
            m.return_value = backend
            yield m, backend

    def test_default_backend_is_vllm(self, mock_vllm_client_class):
        mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("hello world")
        mock_cls.assert_called()
        assert wl.prompt == "hello world"
        backend.wait_for_ready.assert_not_called()
        wl.shutdown_executor(wait=True)

    def test_passes_host_and_port_to_vllm_client(self, mock_vllm_client_class):
        mock_cls, _backend = mock_vllm_client_class
        wl = PromptWorkload(
            "q",
            backend_type=BACKEND_TYPE.VLLM,
            host="vllm.internal",
            port=9090,
        )
        mock_cls.assert_called()
        wl.shutdown_executor(wait=True)

    def test_start_waits_for_backend_and_dispatches_completion(self, mock_vllm_client_class):
        mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("run this")
        wl.start()

        backend.wait_for_ready.assert_called_once_with(timeout=30)
        backend.complete.assert_called_once()
        fut = wl._completion_fut
        assert fut is not None
        fut.result(timeout=10.0)
        assert wl.status == WorkloadStatus.COMPLETED
        wl.shutdown_executor(wait=True)

    def test_stop_cancels_future(self, mock_vllm_client_class):
        _mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("x")
        fake_fut = MagicMock()
        with patch.object(wl, "submit_background", return_value=fake_fut):
            wl.start()
            wl.stop()
        fake_fut.cancel.assert_called_once()
        assert wl.status == WorkloadStatus.STOPPED
        wl.shutdown_executor(wait=True)

    def test_get_result_returns_inference_text_after_completion(self, mock_vllm_client_class):
        mock_cls, backend = mock_vllm_client_class
        backend.complete = MagicMock(return_value=InferenceResult(success=True, text="final text"))
        wl = PromptWorkload("prompt")
        wl.start()
        wl._completion_fut.result(timeout=10.0)
        assert wl.status == WorkloadStatus.COMPLETED
        assert wl.get_result().result.text == "final text"
        assert wl.get_result().status == WorkloadStatus.COMPLETED
        assert wl.get_result().start_time is not None
        assert wl.get_result().end_time is not None
        assert wl.get_result().runtime is not None
        wl.shutdown_executor(wait=True)

    def test_second_start_raises_when_already_running(self, mock_vllm_client_class):
        _mock_cls, backend = mock_vllm_client_class
        block = threading.Event()

        def blocking_complete(_prompt):
            block.wait(timeout=60.0)
            return InferenceResult(success=True, text="ok")

        backend.complete = MagicMock(side_effect=blocking_complete)
        wl = PromptWorkload("x")
        wl.start()
        assert wl.status == WorkloadStatus.RUNNING
        with pytest.raises(
            RuntimeError,
            match="Prompt workload already running",
        ):
            wl.start()
        block.set()
        wl._completion_fut.result(timeout=10.0)
        wl.shutdown_executor(wait=True)
