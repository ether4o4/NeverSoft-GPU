"""Execution runtime for validated NeverGPU command streams."""

from .commands import Command, CommandValidationError, Opcode, SandboxPolicy, validate_command
from .device import Device


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
        op = command.opcode
        args = command.args

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
            self._total_memory -= self._buffers[handle].size
            del self._buffers[handle]
            return None
        if op is Opcode.UPLOAD:
            buffer = self._require_handle(args.get("handle"))
            data = bytes(args["data"])
            offset = args.get("offset", 0)
            buffer.write(offset, data)
            return len(data)
        if op is Opcode.DOWNLOAD:
            buffer = self._require_handle(args.get("handle"))
            size = args.get("size", buffer.size - args.get("offset", 0))
            return buffer.read(args.get("offset", 0), size)
        if op is Opcode.DISPATCH:
            kernel = args.get("kernel")
            return self.device.dispatch(kernel, args["work_items"], workers=args.get("workers"))
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
