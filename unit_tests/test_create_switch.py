# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

import pytest

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import NvidiaCumulusSwitch
from production_test_framework.switch.switch_status import create_switch


def test_create_switch_builds_arista_driver() -> None:
    switch = create_switch("arista-eos", "h", "u", "p")
    assert isinstance(switch, AristaEosSwitch)


def test_create_switch_builds_nvidia_driver() -> None:
    switch = create_switch("nvidia-cumulus", "h", "u", "p")
    assert isinstance(switch, NvidiaCumulusSwitch)


def test_create_switch_rejects_unknown_type() -> None:
    with pytest.raises(SystemExit):
        create_switch("bogus", "h", "u", "p")
