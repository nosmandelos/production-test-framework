# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""NVIDIA Spectrum switch driver via Cumulus Linux NVUE (read-only)."""

import logging
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.models import NetworkSwitchConfig, NetworkSwitchStatus, Port, Vlan
from production_test_framework.switch.network_switch import NetworkSwitch
from production_test_framework.switch.nvidia.nvue_paths import (
    BRIDGE_DOMAIN,
    BRIDGE_DOMAIN_VLANS_PATH,
    FIRMWARE_PATH,
    INTERFACES_PATH,
    PLATFORM_PATH,
    SYSTEM_PATH,
    interface_path,
)
from production_test_framework.switch.port_sort import port_id_sort_key

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# OpenAPI getInterfaces view=description: admin-status, oper-status, description
VIEW_DESCRIPTION = "description"


class NvidiaCumulusSwitch(NetworkSwitch):
    """Read-only NVUE client for Cumulus Linux (e.g. Spectrum-5610)."""

    def __init__(self, config: NetworkSwitchConfig) -> None:
        super().__init__(config)
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing NvidiaCumulusSwitch with config: {config}")
        self._api_root = "/nvue_v1"
        self._base_url = f"https://{config.host}:{config.port}{self._api_root}"
        self._interfaces_config_cache: dict[str, Any] | None = None
        self._interfaces_applied_cache: dict[str, Any] | None = None

    @property
    def status(self) -> NetworkSwitchStatus:
        system = self._run_api_call(SYSTEM_PATH)
        platform = self._run_api_call(PLATFORM_PATH)
        firmware = self._run_api_call(FIRMWARE_PATH)
        return self._parse_system_status(system, platform, firmware)

    @property
    def ports(self) -> list[Port]:
        """All interfaces (OpenAPI operationId: getInterfaces, view=description)."""
        interfaces = self._run_api_call(
            INTERFACES_PATH,
            params={"view": VIEW_DESCRIPTION},
        )
        return self._parse_ports(interfaces)

    @property
    def vlans(self) -> list[Vlan]:
        """Configured VLANs on the bridge domain (OpenAPI operationId: getBridgeDomainVlans)."""
        self.refresh()
        vlan_configs = self._run_api_call(BRIDGE_DOMAIN_VLANS_PATH)
        membership = self._vlan_membership_by_id()
        return self._parse_vlans(vlan_configs, membership)

    def port(self, port_id: str) -> Port:
        """Single interface (OpenAPI operationId: getInterface, view=description)."""
        interface = self._run_api_call(
            interface_path(port_id),
            params={"view": VIEW_DESCRIPTION},
        )
        return self._parse_port(port_id, interface)

    def vlan(self, vlan_id: str) -> Vlan:
        """Single VLAN with member ports (OpenAPI operationId: getBridgeDomainVlan)."""
        self.refresh()
        vlan_configs = self._run_api_call(BRIDGE_DOMAIN_VLANS_PATH)
        if vlan_id not in vlan_configs:
            raise SwitchAPIError(f"VLAN {vlan_id} not found on bridge domain {BRIDGE_DOMAIN}")
        membership = self._vlan_membership_by_id()
        return self._parse_vlan(vlan_id, membership.get(vlan_id, []))

    def refresh(self) -> None:
        """Drop cached interface data so the next read re-fetches from the switch."""
        self._interfaces_config_cache = None
        self._interfaces_applied_cache = None

    def _interfaces_config(self) -> dict[str, Any]:
        """Operational interface state (oper-status, stats)."""
        if self._interfaces_config_cache is None:
            self._interfaces_config_cache = self._run_api_call(INTERFACES_PATH)
        return self._interfaces_config_cache

    def _interfaces_applied_config(self) -> dict[str, Any]:
        """Applied interface config carries bridge VLAN assignment."""
        if self._interfaces_applied_cache is None:
            self._interfaces_applied_cache = self._run_api_call(INTERFACES_PATH, params={"rev": "applied"})
        return self._interfaces_applied_cache

    def _vlan_membership_by_id(self) -> dict[str, list[str]]:
        """Map VLAN ID to interface names assigned on the configured bridge domain."""
        membership: dict[str, list[str]] = {}
        for interface_id, body in self._interfaces_applied_config().items():
            if not isinstance(body, dict):
                continue
            for vid in self._interface_vlan_ids(body, BRIDGE_DOMAIN):
                membership.setdefault(vid, []).append(interface_id)
        for vid in membership:
            membership[vid] = sorted(membership[vid])
        return membership

    @staticmethod
    def _interface_vlan_ids(interface_body: dict[str, Any], bridge_domain: str) -> list[str]:
        bridge = interface_body.get("bridge")
        if not isinstance(bridge, dict):
            return []
        domain_cfg = bridge.get("domain")
        if not isinstance(domain_cfg, dict):
            return []
        domain_body = domain_cfg.get(bridge_domain)
        if not isinstance(domain_body, dict):
            return []
        vlan_cfg = domain_body.get("vlan")
        if not isinstance(vlan_cfg, dict):
            return []
        return list(vlan_cfg.keys())

    def _member_ports(self, port_ids: list[str]) -> tuple[Port, ...]:
        interfaces = self._interfaces_config()
        ports: list[Port] = []
        for port_id in port_ids:
            body = interfaces.get(port_id)
            if isinstance(body, dict):
                ports.append(self._parse_port(port_id, body))
        return tuple(ports)

    def _parse_vlan(self, vlan_id: str, member_port_ids: list[str]) -> Vlan:
        return Vlan(id=vlan_id, ports=self._member_ports(member_port_ids))

    def _parse_vlans(self, vlan_configs: dict[str, Any], membership: dict[str, list[str]]) -> list[Vlan]:
        vlans = [self._parse_vlan(vid, membership.get(vid, [])) for vid in vlan_configs]
        return sorted(vlans, key=lambda vlan: int(vlan.id) if vlan.id.isdigit() else vlan.id)

    def _run_api_call(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        basic = HTTPBasicAuth(self._config.username, self._config.password)
        request_params = dict(params) if params else None

        self._logger.debug(f"NVUE GET {path} params={request_params}")
        response = requests.get(
            url,
            auth=basic,
            params=request_params,
            verify=self._config.verify_tls,
            timeout=30,
        )

        if response.status_code != 200:
            self._logger.error(f"API call failed: {response.status_code} {response.text}")
            raise SwitchAPIError(f"API call failed: {response.status_code} {response.text}")

        result = response.json()
        if not isinstance(result, dict):
            raise SwitchAPIError(f"unexpected NVUE response type for {path}: {type(result).__name__}")
        self._logger.debug(f"NVUE response: {result}")
        return result

    @staticmethod
    def _status_to_bool(status: str | None) -> bool | None:
        if status is None:
            return None
        if status == "up":
            return True
        if status == "down":
            return False
        return None

    def _parse_port(self, interface_id: str, body: dict[str, Any]) -> Port:
        link = body.get("link") if isinstance(body.get("link"), dict) else {}
        admin_status = link.get("admin-status")
        oper_status = link.get("oper-status")
        if admin_status is None and isinstance(link.get("state"), dict):
            state = link["state"]
            if "up" in state:
                admin_status = "up"
            elif "down" in state:
                admin_status = "down"
        description = body.get("description")
        if description == "":
            description = None
        return Port(
            id=interface_id,
            admin_up=self._status_to_bool(admin_status if isinstance(admin_status, str) else None),
            oper_up=self._status_to_bool(oper_status if isinstance(oper_status, str) else None),
            description=description if isinstance(description, str) else None,
        )

    def _parse_ports(self, interfaces: dict[str, Any]) -> list[Port]:
        ports = [self._parse_port(interface_id, body) for interface_id, body in interfaces.items()]
        return sorted(ports, key=lambda port: port_id_sort_key(port.id))

    def _parse_system_status(self, system: dict, platform: dict, firmware: dict) -> NetworkSwitchStatus:
        return NetworkSwitchStatus(
            uptime=system["uptime"],
            hostname=system["hostname"],
            model=platform["product-name"],
            serial_number=platform["serial-number"],
            firmware_version=firmware["Spectrum-4"]["actual-firmware"],
            asic_model=platform["asic-model"],
            software_version=system["version"]["image"],
        )
