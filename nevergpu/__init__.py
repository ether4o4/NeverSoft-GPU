"""NeverGPU: backend-independent sandboxed virtual compute device."""

from .device import Device, DeviceInfo
from .memory import Buffer

__all__ = ["Buffer", "Device", "DeviceInfo"]
