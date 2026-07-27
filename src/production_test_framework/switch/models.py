# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NetworkSwitchConfig:
    host: str
    username: str
    password: str
    port: int = 443
    verify_tls: bool = True
    bridge_domain: str = "br_default"


@dataclass(frozen=True)
class Port:
    id: str
    admin_up: bool | None = None
    oper_up: bool | None = None
    description: str | None = None


@dataclass(frozen=True)
class Vlan:
    id: str
    ports: tuple[Port, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LldpNeighbor:
    """An LLDP neighbor advertising a MAC-address chassis id on a switch port."""

    interface: str
    switch_port: int
    chassis_mac: str


@dataclass(frozen=True)
class SwitchProcess:
    name: str
    pid: int
    status: str
    start_time: str
    end_time: str
    cpu_usage: float
    memory_usage: float


@dataclass(frozen=True)
class NetworkSwitchStatus:
    uptime: str
    hostname: str
    asic_model: str
    model: str
    serial_number: str
    firmware_version: str
    software_version: str
