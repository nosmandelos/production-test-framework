# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for config module."""

import os
from unittest.mock import patch

import pytest

from production_test_framework.config import LGTMConfig


class TestLGTMConfig:
    """Tests for LGTMConfig."""

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = LGTMConfig.from_env()
            assert config.host == "localhost"
            assert config.ansible_remote_user == ""

    def test_from_env_with_env_vars(self):
        with patch.dict(
            os.environ,
            {"REMOTE_HOST": "myhost", "ANSIBLE_REMOTE_USER": "user"},
            clear=False,
        ):
            config = LGTMConfig.from_env()
            assert config.host == "myhost"
            assert config.ansible_remote_user == "user"

    def test_validate_ssh_config_false_when_empty(self):
        config = LGTMConfig(host="x", ansible_remote_user="")
        assert config.validate_ssh_config() is False

    def test_validate_ssh_config_true_when_user_set(self):
        config = LGTMConfig(host="x", ansible_remote_user="user")
        assert config.validate_ssh_config() is True

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "LocalHost"])
    def test_validate_ssh_config_true_for_localhost_without_user(self, host):
        config = LGTMConfig(host=host, ansible_remote_user="")
        assert config.validate_ssh_config() is True
