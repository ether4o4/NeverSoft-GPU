"""Execution runtime for validated NeverGPU command streams."""

from .commands import Command, CommandValidationError, Opcode, SandboxPolicy, validate_command
from .cpu_kernel import execute_kernel
from .device import Device
from .ir import Kernel


class Runtime:
    """Translate virtual GPU commands into backend operations."""

    def __init__(self, device: Device | None = None, policy: SandboxPolicy | None = None) -> None:
        self.device = device or Device()
        self.policy = policy or SandboxPolicy()
        self._buffers: dict[int, object] = {}
        self._next_handle = 1
        self._total_memory = 0

    def execute(self, command: Command):
        validate_command(command, self.policy)
        op, args = command.opcode, command.args
        if op is Opcode.DEVICE_INFO:
            return self.device.info
        if op is Opcode.ALLOC:
            if len(self._buffers) >= self.policy.max_buffers:
                raise CommandValidationError("maximum buffer count reached")
            size = args["size"]
            if self._total_memory + size > self.policy.max_total_memory:
                raise CommandValidationError("total sandbox memory limit exceeded")
            handle = self._next_handle
            self._next_handle += 1
            self._buffers[handle] = self.device.allocate(size)
            self._total_memory += size
            return handle
        if op is Opcode.FREE:
            handle = self._require_handle(args.get("handle"))
            buffer = self._buffers.pop(handle)
            self._total_memory -= buffer.size
            return None
        if op is Opcode.UPLOAD:
            buffer = self._require_handle(args.get("handle"))
            data = bytes(args["data"])
            buffer.write(data, args.get("offset", 0))
            return len(data)
        if op is Opcode.DOWNLOAD:
            buffer = self._require_handle(args.get("handle"))
            return buffer.read(args.get("size"), args.get("offset", 0))
        if op is Opcode.DISPATCH:
            kernel = args.get("kernel")
            if not isinstance(kernel, Kernel):
                raise CommandValidationError("DISPATCH requires a NeverGPU Kernel")
            handles = set()
            for instruction in kernel.instructions:
                handles.update(h for h in (instruction.dst, instruction.src_a, instruction.src_b) if h is not None)
            buffers = {h: self._require_handle(h)._data for h in handles}
            return self.device.dispatch(lambda index: execute_kernel(kernel, buffers, index), args["work_items"], workers=args.get("workers"))
        if op is Opcode.WAIT:
            return None
        raise CommandValidationError(f"unsupported opcode: {op}")

    def _require_handle(self, handle: int):
        if not isinstance(handle, int) or handle not in self._buffers:
            raise CommandValidationError("invalid buffer handle")
        return self._buffers[handle]

    def close(self) -> None:
        self.device.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
