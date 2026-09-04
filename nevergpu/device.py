"""NeverGPU virtual device and reference CPU backend."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from .memory import Buffer


@dataclass(frozen=True)
class DeviceInfo:
    name: str = "NeverGPU CPU Reference"
    backend: str = "cpu"
    max_buffer_size: int = 64 * 1024 * 1024
    max_workers: int = 32


class Device:
    """Minimal virtual compute device.

    v0.1 exposes memory plus a deliberately small kernel-dispatch primitive.
    Kernels are ordinary Python callables for the reference backend; the
    public abstraction is designed so accelerated backends can replace it.
    """

    def __init__(self, info: DeviceInfo | None = None) -> None:
        self.info = info or DeviceInfo()
        self._executor = ThreadPoolExecutor(max_workers=self.info.max_workers)

    def allocate(self, size: int) -> Buffer:
        if size > self.info.max_buffer_size:
            raise ValueError("requested buffer exceeds device limit")
        return Buffer.allocate(size)

    def dispatch(self, kernel, work_items: int, *, workers: int | None = None):
        if not callable(kernel):
            raise TypeError("kernel must be callable")
        if not isinstance(work_items, int) or work_items < 0:
            raise ValueError("work_items must be a non-negative integer")
        if work_items == 0:
            return []

        worker_count = min(workers or self.info.max_workers, work_items)
        chunks = [[] for _ in range(worker_count)]
        for index in range(work_items):
            chunks[index % worker_count].append(index)

        futures = [self._executor.submit(self._run_chunk, kernel, chunk) for chunk in chunks]
        results = []
        for future in futures:
            results.extend(future.result())
        return results

    @staticmethod
    def _run_chunk(kernel, chunk):
        return [kernel(index) for index in chunk]

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
