"""Small, serializable compute IR used by the NeverGPU command layer."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Op(str, Enum):
    ADD = "ADD"
    MUL = "MUL"
    MAD = "MAD"
    COPY = "COPY"
    FILL = "FILL"


@dataclass(frozen=True)
class Instruction:
    op: Op
    dst: int
    src_a: int | None = None
    src_b: int | None = None
    scalar: int | float | None = None


@dataclass(frozen=True)
class Kernel:
    """A deliberately tiny kernel program, safe to validate and serialize."""
    instructions: tuple[Instruction, ...]

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ValueError("kernel must contain at least one instruction")
        for instruction in self.instructions:
            if not isinstance(instruction, Instruction):
                raise TypeError("kernel instructions must be Instruction values")

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "instructions": [
            {"op": i.op.value, "dst": i.dst, "src_a": i.src_a,
             "src_b": i.src_b, "scalar": i.scalar}
            for i in self.instructions
        ]}


def vector_add(dst: int, a: int, b: int) -> Kernel:
    return Kernel((Instruction(Op.ADD, dst=dst, src_a=a, src_b=b),))


def vector_mul(dst: int, a: int, b: int) -> Kernel:
    return Kernel((Instruction(Op.MUL, dst=dst, src_a=a, src_b=b),))
