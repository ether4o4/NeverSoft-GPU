import pytest

from nevergpu.commands import CommandValidationError, Opcode, SandboxPolicy, command
from nevergpu.ir import vector_add
from nevergpu.runtime import Runtime


def test_allocate_upload_download_free():
    with Runtime() as runtime:
        handle = runtime.execute(command(Opcode.ALLOC, size=16))
        assert runtime.execute(command(Opcode.UPLOAD, handle=handle, data=b"hello")) == 5
        assert runtime.execute(command(Opcode.DOWNLOAD, handle=handle, size=5)) == b"hello"
        runtime.execute(command(Opcode.FREE, handle=handle))


def test_free_releases_memory_and_invalidates_handle():
    policy = SandboxPolicy(max_total_memory=16)
    with Runtime(policy=policy) as runtime:
        handle = runtime.execute(command(Opcode.ALLOC, size=16))
        runtime.execute(command(Opcode.FREE, handle=handle))
        with pytest.raises(CommandValidationError):
            runtime.execute(command(Opcode.DOWNLOAD, handle=handle, size=1))
        new_handle = runtime.execute(command(Opcode.ALLOC, size=16))
        assert new_handle != handle


def test_invalid_handle_rejected():
    with Runtime() as runtime:
        with pytest.raises(CommandValidationError):
            runtime.execute(command(Opcode.DOWNLOAD, handle=999, size=1))


def test_dispatch_limit_enforced():
    policy = SandboxPolicy(max_work_items=10)
    with Runtime(policy=policy) as runtime:
        with pytest.raises(CommandValidationError):
            runtime.execute(command(Opcode.DISPATCH, kernel=vector_add(1, 2, 3), work_items=11))


def test_kernel_dispatch():
    with Runtime() as runtime:
        a = runtime.execute(command(Opcode.ALLOC, size=4))
        b = runtime.execute(command(Opcode.ALLOC, size=4))
        dst = runtime.execute(command(Opcode.ALLOC, size=4))
        runtime.execute(command(Opcode.UPLOAD, handle=a, data=bytes([1, 2, 3, 4])))
        runtime.execute(command(Opcode.UPLOAD, handle=b, data=bytes([10, 20, 30, 40])))
        runtime.execute(command(Opcode.DISPATCH, kernel=vector_add(dst, a, b), work_items=4))
        assert runtime.execute(command(Opcode.DOWNLOAD, handle=dst, size=4)) == bytes([11, 22, 33, 44])


def test_device_info():
    with Runtime() as runtime:
        info = runtime.execute(command(Opcode.DEVICE_INFO))
        assert info.backend == "cpu"
