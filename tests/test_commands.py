import pytest

from nevergpu.commands import CommandValidationError, Opcode, SandboxPolicy, command
from nevergpu.runtime import Runtime


def test_allocate_upload_download_free():
    with Runtime() as runtime:
        handle = runtime.execute(command(Opcode.ALLOC, size=16))
        assert runtime.execute(command(Opcode.UPLOAD, handle=handle, data=b"hello")) == 5
        assert runtime.execute(command(Opcode.DOWNLOAD, handle=handle, size=5)) == b"hello"
        runtime.execute(command(Opcode.FREE, handle=handle))


def test_invalid_handle_rejected():
    with Runtime() as runtime:
        with pytest.raises(CommandValidationError):
            runtime.execute(command(Opcode.DOWNLOAD, handle=999, size=1))


def test_dispatch_limit_enforced():
    policy = SandboxPolicy(max_work_items=10)
    with Runtime(policy=policy) as runtime:
        with pytest.raises(CommandValidationError):
            runtime.execute(command(Opcode.DISPATCH, kernel=lambda i: i, work_items=11))


def test_device_info():
    with Runtime() as runtime:
        info = runtime.execute(command(Opcode.DEVICE_INFO))
        assert info.backend == "cpu"
