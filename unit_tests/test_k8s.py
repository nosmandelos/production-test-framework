# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for k8s module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from production_test_framework.config import LGTMConfig
from production_test_framework.k8s import (
    Node,
    Pod,
    KubectlPortForwarder,
    KubernetesClient,
    LocalKubectlPortForwarder,
    LocalPortForward,
)
from production_test_framework.ssh import CommandResult


class TestNode:
    """Tests for Node dataclass."""

    def test_is_ready_true(self):
        node = Node(name="node1", status="Ready", roles="control-plane", version="v1.28")
        assert node.is_ready is True

    def test_is_ready_false(self):
        node = Node(name="node1", status="NotReady", roles="control-plane", version="v1.28")
        assert node.is_ready is False


class TestPod:
    """Tests for Pod dataclass."""

    def test_is_running_true(self):
        pod = Pod("pod1", "default", "Running", "1/1", 0, "5m")
        assert pod.is_running is True

    def test_is_running_false(self):
        pod = Pod("pod1", "default", "Pending", "0/1", 0, "1m")
        assert pod.is_running is False

    def test_is_completed_true(self):
        pod = Pod("pod1", "default", "Completed", "1/1", 0, "10m")
        assert pod.is_completed is True

    def test_is_ready_true(self):
        pod = Pod("pod1", "default", "Running", "2/2", 0, "5m")
        assert pod.is_ready is True

    def test_is_ready_false_no_slash(self):
        pod = Pod("pod1", "default", "Running", "1", 0, "5m")
        assert pod.is_ready is False

    def test_is_ready_false_partial(self):
        pod = Pod("pod1", "default", "Running", "1/2", 0, "5m")
        assert pod.is_ready is False


class TestKubernetesClient:
    """Tests for KubernetesClient with mocked SSH."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.run_kubectl = MagicMock()
        return ssh

    @pytest.fixture
    def k8s_client(self, lgtm_config, mock_ssh):
        return KubernetesClient(lgtm_config, ssh=mock_ssh)

    def test_get_nodes_parses_output(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="node1   Ready   control-plane   1.28   192.168.1.1\n"
            "node2   Ready   <none>   1.28   192.168.1.2\n",
            stderr="",
        )

        nodes, result = k8s_client.get_nodes()

        assert result.success is True
        assert len(nodes) == 2
        assert nodes[0].name == "node1"
        assert nodes[0].status == "Ready"
        assert nodes[0].internal_ip == None
        assert nodes[1].name == "node2"

    def test_get_node_count(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="node1   Ready   control-plane   1.28   192.168.1.1\n",
            stderr="",
        )
        assert k8s_client.get_node_count() == 1

    def test_all_nodes_ready_true(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="node1   Ready   control-plane   1.28   192.168.1.1\n",
            stderr="",
        )
        assert k8s_client.all_nodes_ready() is True

    def test_all_nodes_ready_false_when_not_ready(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="node1   NotReady   control-plane   1.28   192.168.1.1\n",
            stderr="",
        )
        assert k8s_client.all_nodes_ready() is False

    def test_get_pods_all_namespaces(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="default   pod1  1/1   Running   0   5m\n"
            "kube-system   coredns  2/2   Running   0   10m\n",
            stderr="",
        )

        pods, result = k8s_client.get_pods(all_namespaces=True)

        assert result.success is True
        assert len(pods) == 2
        assert pods[0].namespace == "default" and pods[0].name == "pod1"
        assert pods[1].namespace == "kube-system" and pods[1].name == "coredns"
        mock_ssh.run_kubectl.assert_called_with("get pods -A --no-headers", timeout=60, stdin_data=None)

    def test_get_pods_single_namespace(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pod1  1/1   Running   0   5m\n",
            stderr="",
        )

        pods, _ = k8s_client.get_pods(namespace="default")

        assert len(pods) == 1
        assert pods[0].namespace == "default"
        assert pods[0].name == "pod1"
        mock_ssh.run_kubectl.assert_called_with("get pods -n default --no-headers", timeout=60, stdin_data=None)

    def test_get_pods_in_namespace(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pod1  1/1   Running   0   5m\n",
            stderr="",
        )
        pods = k8s_client.get_pods_in_namespace("default")
        assert len(pods) == 1
        assert pods[0].name == "pod1"

    def test_all_pods_running(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pod1  1/1   Running   0   5m\npod2  1/1   Running   0   5m\n",
            stderr="",
        )
        assert k8s_client.all_pods_running("default") is True

    def test_all_pods_ready_includes_completed(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="job-pod  1/1   Completed   0   5m\n",
            stderr="",
        )
        assert k8s_client.all_pods_ready("default") is True

    def test_get_namespaces(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="default\nkube-system\nmonitoring\n",
            stderr="",
        )
        namespaces, result = k8s_client.get_namespaces()
        assert result.success is True
        assert set(namespaces) == {"default", "kube-system", "monitoring"}

    def test_namespace_exists_true(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        assert k8s_client.namespace_exists("default") is True

    def test_namespace_exists_false(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="")
        assert k8s_client.namespace_exists("missing") is False

    def test_all_namespaces_exist(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="default\nkube-system\n",
            stderr="",
        )
        all_exist, missing = k8s_client.all_namespaces_exist(["default", "kube-system", "other"])
        assert all_exist is False
        assert missing == ["other"]

    def test_get_pvcs(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pvc-1\npvc-2\n",
            stderr="",
        )
        pvcs, result = k8s_client.get_pvcs("default")
        assert result.success is True
        assert pvcs == ["pvc-1", "pvc-2"]

    def test_namespace_has_pvcs_true(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pvc-1\n",
            stderr="",
        )
        assert k8s_client.namespace_has_pvcs("default") is True

    def test_service_exists(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        assert k8s_client.service_exists("grafana", "default") is True

    def test_get_service_port(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="3000",
            stderr="",
        )
        assert k8s_client.get_service_port("grafana", "default") == 3000

    def test_get_service_port_invalid_returns_none(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="",
            stderr="",
        )
        assert k8s_client.get_service_port("svc", "default") is None

    def test_delete_service(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        assert k8s_client.delete_service("grafana", "default") is True

    def test_wait_for_pods_ready(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        assert k8s_client.wait_for_pods_ready("default", "app=grafana", timeout=60) is True
        call_args = mock_ssh.run_kubectl.call_args[0][0]
        assert "wait" in call_args
        assert "condition=Ready" in call_args
        assert "app=grafana" in call_args
        assert "timeout=60s" in call_args

    def test_apply_manifest_file(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        manifest = Path("/tmp/manifest.yaml")
        with patch.object(Path, "read_text", return_value="apiVersion: v1\nkind: ConfigMap\n"):
            result = k8s_client.apply_manifest_file(manifest, "default")
        assert result is True
        mock_ssh.run_kubectl.assert_called_once()
        assert mock_ssh.run_kubectl.call_args[1]["stdin_data"] == "apiVersion: v1\nkind: ConfigMap\n"


class TestKubernetesClientLocalhost:
    """KubernetesClient uses local kubectl when host is loopback."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.run_kubectl = MagicMock()
        return ssh

    @pytest.fixture
    def localhost_config(self):
        return LGTMConfig(host="localhost", ansible_remote_user="u")

    def test_localhost_uses_subprocess_not_ssh(self, localhost_config, mock_ssh):
        client = KubernetesClient(localhost_config, ssh=mock_ssh)
        with patch("production_test_framework.helper.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="node1   Ready   control-plane   1.28   192.168.1.1\n",
                stderr="",
            )
            nodes, result = client.get_nodes()
            assert result.success is True
            assert len(nodes) == 1
            mock_ssh.run_kubectl.assert_not_called()
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "kubectl"
            assert cmd[1] == "--kubeconfig"
            assert "get nodes -o wide --no-headers" in " ".join(cmd)

    def test_127_0_0_1_uses_subprocess(self, mock_ssh):
        cfg = LGTMConfig(host="127.0.0.1", ansible_remote_user="u")
        client = KubernetesClient(cfg, ssh=mock_ssh)
        with patch("production_test_framework.helper.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            client.get_namespaces()
            mock_ssh.run_kubectl.assert_not_called()
            mock_run.assert_called_once()


class TestKubectlPortForwarder:
    """Tests for the SSH-based KubectlPortForwarder with a mocked SSHExecutor."""

    @staticmethod
    def _ssh_mock(listening_port: int | None) -> MagicMock:
        """SSHExecutor stub whose `ss -ltn` reports listening_port bound, if any."""
        ssh = MagicMock()

        def run(command: str, *args, **kwargs) -> CommandResult:
            if command.startswith("ss -ltn"):
                stdout = f"LISTEN 0 4096 127.0.0.1:{listening_port} 0.0.0.0:*" if listening_port else ""
                return CommandResult(returncode=0, stdout=stdout, stderr="")
            if command.startswith("tail"):
                return CommandResult(returncode=0, stdout="error: services 'mimir-gateway' not found", stderr="")
            return CommandResult(returncode=0, stdout="12345", stderr="")

        ssh.run.side_effect = run
        return ssh

    def test_start_service_tunnel_waits_for_remote_bind(self):
        ssh = self._ssh_mock(listening_port=9009)
        forwarder = KubectlPortForwarder(ssh)

        with patch.object(forwarder, "_start_ssh_tunnel", return_value=True) as mock_tunnel:
            assert forwarder.start_service_tunnel(
                local_port=9009,
                service_name="mimir-gateway",
                service_port=80,
                namespace="lgtma",
                use_sudo=False,
            )

        commands = [c.args[0] for c in ssh.run.call_args_list]
        kubectl = next(c for c in commands if "port-forward" in c)
        assert "kubectl -n lgtma port-forward svc/mimir-gateway 9009:80" in kubectl
        assert not kubectl.startswith("sudo"), "use_sudo=False must not prepend sudo"
        assert any(c.startswith("ss -ltn") for c in commands), "must verify the remote port is bound"
        mock_tunnel.assert_called_once()
        assert 12345 in forwarder._kubectl_pids

    @patch("production_test_framework.helper.time.sleep")
    def test_start_service_tunnel_fails_when_remote_never_binds(self, mock_sleep):
        # kubectl exits immediately (wrong namespace, missing service): nothing
        # ever listens, so the tunnel must be reported as failed rather than
        # handed back broken.
        ssh = self._ssh_mock(listening_port=None)
        forwarder = KubectlPortForwarder(ssh)

        with patch.object(forwarder, "_start_ssh_tunnel", return_value=True) as mock_tunnel:
            assert not forwarder.start_service_tunnel(
                local_port=9009,
                service_name="mimir-gateway",
                service_port=80,
                namespace="mosaic",
                ready_timeout=1.0,
            )

        mock_tunnel.assert_not_called()

    def test_start_service_tunnel_proceeds_when_ss_is_unavailable(self):
        # A host without iproute2 cannot be probed; that must not fail every
        # tunnel, so an unusable probe falls back to the previous behavior.
        ssh = MagicMock()
        ssh.run.side_effect = lambda command, *a, **kw: (
            CommandResult(returncode=127, stdout="", stderr="ss: command not found")
            if command.startswith("ss -ltn")
            else CommandResult(returncode=0, stdout="12345", stderr="")
        )
        forwarder = KubectlPortForwarder(ssh)

        with patch.object(forwarder, "_start_ssh_tunnel", return_value=True) as mock_tunnel:
            assert forwarder.start_service_tunnel(
                local_port=9009,
                service_name="mimir-gateway",
                service_port=80,
                ready_timeout=1.0,
            )

        mock_tunnel.assert_called_once()

    def test_start_service_tunnel_can_skip_the_readiness_wait(self):
        ssh = self._ssh_mock(listening_port=None)
        forwarder = KubectlPortForwarder(ssh)

        with patch.object(forwarder, "_start_ssh_tunnel", return_value=True):
            assert forwarder.start_service_tunnel(
                local_port=9009,
                service_name="mimir-gateway",
                service_port=80,
                wait_ready=False,
            )

        assert not any(c.args[0].startswith("ss -ltn") for c in ssh.run.call_args_list)


class TestLocalKubectlPortForwarder:
    """Tests for LocalKubectlPortForwarder with mocked subprocess."""

    @patch("production_test_framework.k8s.subprocess.Popen")
    def test_start_service_tunnel_builds_correct_command(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        with patch("production_test_framework.k8s.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            mock_sock.connect_ex.return_value = 0

            forwarder = LocalKubectlPortForwarder(namespace="default")
            result = forwarder.start_service_tunnel(
                local_port=3000,
                service_name="grafana",
                service_port=80,
                wait_ready=True,
                ready_timeout=1.0,
            )

            assert result is True
            call_args = mock_popen.call_args[0][0]
            assert "kubectl" in call_args
            assert "-n" in call_args
            assert "default" in call_args
            assert "port-forward" in call_args
            assert "svc/grafana" in call_args
            assert "3000:80" in call_args

    def test_stop_service_tunnel_removes_forward(self):
        forwarder = LocalKubectlPortForwarder()
        mock_proc = MagicMock()
        forwarder._forwards.append(
            LocalPortForward(process=mock_proc, local_port=3000, service="grafana", namespace="default")
        )
        result = forwarder.stop_service_tunnel("grafana")
        assert result is True
        mock_proc.terminate.assert_called_once()
        assert len(forwarder._forwards) == 0

    def test_stop_service_tunnel_not_found(self):
        forwarder = LocalKubectlPortForwarder()
        assert forwarder.stop_service_tunnel("nonexistent") is False

    def test_is_running(self):
        forwarder = LocalKubectlPortForwarder()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        forwarder._forwards.append(
            LocalPortForward(process=mock_proc, local_port=3000, service="grafana", namespace="default")
        )
        assert forwarder.is_running("grafana") is True
        mock_proc.poll.return_value = 1
        assert forwarder.is_running("grafana") is False

    def test_get_local_port(self):
        forwarder = LocalKubectlPortForwarder()
        forwarder._forwards.append(
            LocalPortForward(
                process=MagicMock(),
                local_port=3000,
                service="grafana",
                namespace="default",
            )
        )
        assert forwarder.get_local_port("grafana") == 3000
        assert forwarder.get_local_port("other") is None
