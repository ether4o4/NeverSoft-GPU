"""Versioned command protocol for the NeverGPU sandbox boundary."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Opcode(str, Enum):
    DEVICE_INFO = "DEVICE_INFO"
    ALLOC = "ALLOC"
    FREE = "FREE"
    UPLOAD = "UPLOAD"
    DOWNLOAD = "DOWNLOAD"
    DISPATCH = "DISPATCH"
    WAIT = "WAIT"


@dataclass(frozen=True)
class Command:
    opcode: Opcode
    args: dict[str, Any]


@dataclass(frozen=True)
class SandboxPolicy:
    max_buffers: int = 256
    max_total_memory: int = 256 * 1024 * 1024
    max_work_items: int = 10_000_000
    max_upload_size: int = 64 * 1024 * 1024


class CommandValidationError(ValueError):
    """Raised when a command violates the virtual device contract."""


def validate_command(command: Command, policy: SandboxPolicy) -> None:
    if not isinstance(command, Command):
        raise TypeError("command must be a Command")
    if not isinstance(command.opcode, Opcode):
        raise CommandValidationError("unknown opcode")
    if not isinstance(command.args, dict):
        raise CommandValidationError("command args must be an object")

    if command.opcode is Opcode.ALLOC:
        size = command.args.get("size")
        if not isinstance(size, int) or size <= 0:
            raise CommandValidationError("ALLOC requires a positive integer size")
        if size > policy.max_total_memory:
            raise CommandValidationError("allocation exceeds sandbox memory limit")

    elif command.opcode is Opcode.UPLOAD:
        data = command.args.get("data")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise CommandValidationError("UPLOAD requires bytes-like data")
        if len(data) > policy.max_upload_size:
            raise CommandValidationError("upload exceeds sandbox limit")

    elif command.opcode is Opcode.DISPATCH:
        work_items = command.args.get("work_items")
        if not isinstance(work_items, int) or work_items < 0:
            raise CommandValidationError("DISPATCH requires non-negative integer work_items")
        if work_items > policy.max_work_items:
            raise CommandValidationError("dispatch exceeds sandbox work-item limit")


def command(opcode: Opcode, **args: Any) -> Command:
    """Convenience constructor for command streams."""
    return Command(opcode=opcode, args=args)
