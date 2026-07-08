# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

from abc import ABC, abstractmethod

from production_test_framework.switch.models import NetworkSwitchConfig, NetworkSwitchStatus, Port, Vlan


class NetworkSwitch(ABC):
    def __init__(self, config: NetworkSwitchConfig) -> None:
        self._config = config

    @property
    @abstractmethod
    def status(self) -> NetworkSwitchStatus:
        """Get the switch status"""
        ...

    @property
    @abstractmethod
    def ports(self) -> list[Port]:
        """Get the ports of the switch"""
        ...

    @property
    @abstractmethod
    def vlans(self) -> list[Vlan]:
        """Get the vlans of the switch"""
        ...

    @abstractmethod
    def port(self, port_id: str) -> Port:
        """Get configuration for a port of the switch."""
        ...

    @abstractmethod
    def vlan(self, vlan_id: str) -> Vlan:
        """Get configuration for a vlan of the switch."""
        ...

    @abstractmethod
    def set_port_admin_state(self, port_id: str, up: bool) -> None:
        """Administratively enable (up=True) or disable (up=False) a port."""
        ...

    @abstractmethod
    def delete_vlan(self, vlan_id: str) -> None:
        """Remove a VLAN from the bridge domain and from every member interface."""
        ...
