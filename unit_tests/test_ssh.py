# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for ssh module."""

import socket
from unittest.mock import MagicMock, patch

from production_test_framework.ssh import (
    CommandResult,
    SSHExecutor,
    check_ssh_password_login,
    wait_for_ssh_password_login,
)


class TestCommandResult:
    """Tests for CommandResult."""

    def test_success_true_when_zero(self):
        r = CommandResult(returncode=0, stdout="ok", stderr="")
        assert r.success is True
        assert bool(r) is True

    def test_success_false_when_nonzero(self):
        r = CommandResult(returncode=1, stdout="", stderr="error")
        assert r.success is False
        assert bool(r) is False


class TestSSHExecutor:
    """Tests for SSHExecutor with mocked Paramiko."""

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_run_success(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"hello"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        result = executor.run("echo hello")

        assert result.success is True
        assert result.stdout == "hello"
        assert result.returncode == 0
        mock_client.connect.assert_called_once()
        mock_client.exec_command.assert_called_once_with("echo hello", timeout=60)

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_run_connection_error_returns_command_result(self, mock_ssh_cls, lgtm_config):
        mock_ssh_cls.return_value = MagicMock()
        mock_ssh_cls.return_value.connect.side_effect = Exception("Connection refused")

        executor = SSHExecutor(lgtm_config)
        result = executor.run("echo hello")

        assert result.success is False
        assert result.returncode == -1
        assert "Connection refused" in result.stderr

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_run_kubectl_builds_correct_command(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        executor.run_kubectl("get pods -A")

        call_args = mock_client.exec_command.call_args[0][0]
        assert "kubectl" in call_args
        assert "--kubeconfig" in call_args
        assert "get pods -A" in call_args

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_run_with_stdin_data(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        executor.run("cat", stdin_data="hello\n")

        mock_stdin.write.assert_called_once_with("hello\n")
        mock_stdin.flush.assert_called_once()
        mock_stdin.channel.shutdown_write.assert_called_once()

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_check_service_active_true(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"active"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        assert executor.check_service_active("k3s") is True
        call_args = mock_client.exec_command.call_args[0][0]
        assert "systemctl is-active k3s" in call_args

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_check_service_active_false(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"inactive"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        assert executor.check_service_active("k3s") is False

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_file_exists_true(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        assert executor.file_exists("/etc/hosts") is True
        call_args = mock_client.exec_command.call_args[0][0]
        assert "test -f '/etc/hosts'" in call_args

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_read_file_returns_content(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"file content"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        content = executor.read_file("/path/to/file")
        assert content == "file content"

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_read_file_returns_none_on_failure(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b"No such file"
        mock_stdout.channel.recv_exit_status.return_value = 1
        mock_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        assert executor.read_file("/nonexistent") is None

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_run_timeout_returns_command_result(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_client.exec_command.side_effect = socket.timeout()

        executor = SSHExecutor(lgtm_config)
        result = executor.run("sleep 100", timeout=5)

        assert result.success is False
        assert result.returncode == -1
        assert "timed out" in result.stderr

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_close_clears_client(self, mock_ssh_cls, lgtm_config):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)

        executor = SSHExecutor(lgtm_config)
        executor.run("echo ok")
        executor.close()

        mock_client.close.assert_called_once()
        assert executor._client is None
        assert executor._current_host is None


class TestCheckSSHPasswordLogin:
    """Tests for check_ssh_password_login with mocked Paramiko."""

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_login_and_command_success(self, mock_ssh_cls):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"ok"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)

        result = check_ssh_password_login("10.0.0.5", "ubuntu", "pw")

        assert result.success is True
        assert result.stdout == "ok"
        # password auth only: no key/agent lookups
        kwargs = mock_client.connect.call_args.kwargs
        assert kwargs["username"] == "ubuntu"
        assert kwargs["password"] == "pw"
        assert kwargs["allow_agent"] is False
        assert kwargs["look_for_keys"] is False
        mock_client.close.assert_called_once()

    @patch("production_test_framework.ssh.paramiko.SSHClient")
    def test_connection_failure_returns_negative(self, mock_ssh_cls):
        mock_client = MagicMock()
        mock_ssh_cls.return_value = mock_client
        mock_client.connect.side_effect = Exception("auth failed")

        result = check_ssh_password_login("10.0.0.5", "ubuntu", "wrong")

        assert result.success is False
        assert result.returncode == -1
        assert "auth failed" in result.stderr
        mock_client.exec_command.assert_not_called()


class TestWaitForSSHPasswordLogin:
    """Tests for wait_for_ssh_password_login."""

    @patch("production_test_framework.ssh.check_ssh_password_login")
    def test_returns_on_first_success(self, mock_login):
        mock_login.return_value = CommandResult(returncode=0, stdout="", stderr="")

        result = wait_for_ssh_password_login("10.0.0.5", "ubuntu", "pw", timeout=5)

        assert result.success is True
        assert mock_login.call_count == 1

    @patch("production_test_framework.ssh.time.sleep")
    @patch("production_test_framework.ssh.check_ssh_password_login")
    def test_retries_until_success(self, mock_login, mock_sleep):
        mock_login.side_effect = [
            CommandResult(returncode=-1, stdout="", stderr="refused"),
            CommandResult(returncode=0, stdout="", stderr=""),
        ]

        result = wait_for_ssh_password_login("10.0.0.5", "ubuntu", "pw", timeout=30, interval=1)

        assert result.success is True
        assert mock_login.call_count == 2

    @patch("production_test_framework.ssh.time.monotonic")
    @patch("production_test_framework.ssh.time.sleep")
    @patch("production_test_framework.ssh.check_ssh_password_login")
    def test_returns_last_failure_on_timeout(self, mock_login, mock_sleep, mock_monotonic):
        # deadline computed at t=0 (=> 10); one attempt at t=0, then t=100 exits the loop.
        mock_monotonic.side_effect = [0, 0, 100]
        mock_login.return_value = CommandResult(returncode=-1, stdout="", stderr="refused")

        result = wait_for_ssh_password_login("10.0.0.5", "ubuntu", "pw", timeout=10, interval=1)

        assert result.success is False
        assert result.stderr == "refused"
        assert mock_login.call_count == 1
