"""Virtual device memory primitives."""

from dataclasses import dataclass


@dataclass
class Buffer:
    """A bounded block of virtual device memory."""

    size: int
    _data: bytearray

    @classmethod
    def allocate(cls, size: int) -> "Buffer":
        if not isinstance(size, int) or size <= 0:
            raise ValueError("buffer size must be a positive integer")
        return cls(size=size, _data=bytearray(size))

    def write(self, data: bytes, offset: int = 0) -> None:
        end = offset + len(data)
        if offset < 0 or end > self.size:
            raise ValueError("buffer write exceeds allocation")
        self._data[offset:end] = data

    def read(self, size: int | None = None, offset: int = 0) -> bytes:
        if size is None:
            size = self.size - offset
        end = offset + size
        if offset < 0 or size < 0 or end > self.size:
            raise ValueError("buffer read exceeds allocation")
        return bytes(self._data[offset:end])
