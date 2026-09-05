"""Reference CPU interpreter for NeverGPU compute kernels."""

from .ir import Instruction, Kernel, Op


def execute_kernel(kernel: Kernel, buffers: dict[int, bytearray], index: int) -> None:
    """Execute one work item using the restricted NeverGPU IR.

    Buffers are byte-oriented in v0.1. Each instruction operates on one byte
    at the supplied work-item index, making the reference backend deterministic
    and easy to compare with accelerated backends later.
    """
    for instruction in kernel.instructions:
        _execute_instruction(instruction, buffers, index)


def _execute_instruction(i: Instruction, buffers: dict[int, bytearray], index: int) -> None:
    dst = _byte(buffers, i.dst, index)
    if i.op is Op.FILL:
        if i.scalar is None:
            raise ValueError("FILL requires scalar")
        buffers[i.dst][index] = int(i.scalar) & 0xFF
    elif i.op is Op.COPY:
        buffers[i.dst][index] = _byte(buffers, i.src_a, index)
    elif i.op is Op.ADD:
        buffers[i.dst][index] = (_byte(buffers, i.src_a, index) + _byte(buffers, i.src_b, index)) & 0xFF
    elif i.op is Op.MUL:
        buffers[i.dst][index] = (_byte(buffers, i.src_a, index) * _byte(buffers, i.src_b, index)) & 0xFF
    elif i.op is Op.MAD:
        buffers[i.dst][index] = (_byte(buffers, i.src_a, index) * _byte(buffers, i.src_b, index) + int(i.scalar or 0)) & 0xFF
    else:
        raise ValueError(f"unsupported instruction: {i.op}")


def _byte(buffers, handle, index):
    if handle not in buffers:
        raise ValueError("invalid kernel buffer handle")
    data = buffers[handle]
    if not 0 <= index < len(data):
        raise IndexError("kernel index outside buffer")
    return data[index]
