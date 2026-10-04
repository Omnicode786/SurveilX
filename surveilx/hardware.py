import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache

import psutil


@lru_cache(maxsize=1)
def hardware_identity():
    """Inventory, independent of whether an acceleration library is installed."""
    result = {"cpu_name": platform.processor(), "display_adapters": []}
    if os.name == "nt":
        import winreg

        for path, fields in [
            (r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", {"ProcessorNameString": "cpu_name"}),
            (r"HARDWARE\DESCRIPTION\System\BIOS", {"SystemManufacturer": "manufacturer",
                                                       "SystemProductName": "computer_model"}),
        ]:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                    for field, name in fields.items():
                        try:
                            result[name] = winreg.QueryValueEx(key, field)[0]
                        except OSError:
                            pass
            except OSError:
                pass
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}") as root:
                for index in range(winreg.QueryInfoKey(root)[0]):
                    child = winreg.EnumKey(root, index)
                    if child.isdigit():
                        with winreg.OpenKey(root, child) as key:
                            try:
                                result["display_adapters"].append(winreg.QueryValueEx(key, "DriverDesc")[0])
                            except OSError:
                                pass
        except OSError:
            pass
    return result


@dataclass
class Hardware:
    os: str
    architecture: str
    cores: int
    ram_gb: float
    available_gb: float
    cpu_percent: float
    memory_percent: float
    gpu: dict | None
    temperature_c: float | None
    battery_percent: float | None
    plugged_in: bool | None
    identity: dict | None = None

    def json(self):
        return asdict(self)


def probe():
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    gpu = None
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            cells = result.stdout.strip().splitlines()[0].split(",")

            def number(value):
                try:
                    return float(value.strip())
                except ValueError:
                    return None

            gpu = dict(
                zip(
                    ["name", "memory_mb", "used_mb", "utilization", "temperature_c", "power_w"],
                    [cells[0].strip(), *map(number, cells[1:])],
                )
            )
            if gpu.get("temperature_c") is not None and gpu["temperature_c"] <= 0:
                gpu["temperature_c"] = None
        except (OSError, subprocess.TimeoutExpired, IndexError):
            pass
    temperatures = getattr(psutil, "sensors_temperatures", lambda: {})()
    readings = [r.current for group in temperatures.values() for r in group if r.current > 0]
    temp = max(readings) if readings else None
    if gpu and gpu.get("temperature_c") is not None:
        temp = max(temp or 0, gpu["temperature_c"])
    return Hardware(
        platform.system(),
        platform.machine(),
        psutil.cpu_count(logical=False) or 1,
        memory.total / 2**30,
        memory.available / 2**30,
        psutil.cpu_percent(),
        memory.percent,
        gpu,
        temp,
        battery.percent if battery else None,
        battery.power_plugged if battery else None,
        hardware_identity(),
    )


class PowerGovernor:
    """Application compute limits, not OS power plans. Hysteretic, conservative CPU cold start."""

    levels = ["economy", "balanced", "performance"]

    def __init__(self):
        self.index = 0
        self.healthy_ticks = 0
        self.reason = "Conservative startup; collecting measured load"

    def update(self, hardware, latency_ms=0):
        ceiling = 2 if hardware.cores >= 8 and hardware.ram_gb >= 12 else 1
        if hardware.ram_gb < 4:
            ceiling = 0
        pressure = (
            hardware.cpu_percent > 85
            or hardware.memory_percent > 88
            or hardware.available_gb < 0.75
            or (hardware.temperature_c is not None and hardware.temperature_c >= 82)
            or (hardware.plugged_in is False and (hardware.battery_percent or 0) < 25)
            or latency_ms > 700
        )
        if pressure:
            self.index = max(0, self.index - 1)
            self.healthy_ticks = 0
            self.reason = "Reduced workload: load, latency, thermal or battery pressure"
        else:
            self.healthy_ticks += 1
            if self.healthy_ticks >= 10:
                self.index = min(ceiling, self.index + 1)
                self.healthy_ticks = 0
            self.reason = "Hardware capacity and sustained measured headroom"
        self.index = min(self.index, ceiling)
        return {
            "level": self.levels[self.index],
            "reason": self.reason,
            "resolution": [320, 480, 640][self.index],
            "budget_ms": [200, 450, 700][self.index],
            "capture_fps": [5, 8, 12][self.index],
            "watts_target": None,
            "os_power_plan_changed": False,
        }
