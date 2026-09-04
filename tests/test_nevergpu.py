from nevergpu import Device


def test_buffer_round_trip():
    with Device() as gpu:
        buf = gpu.allocate(16)
        buf.write(b"NeverGPU")
        assert buf.read(8) == b"NeverGPU"


def test_parallel_dispatch():
    with Device() as gpu:
        result = gpu.dispatch(lambda i: i * i, 1000)
        assert result == [i * i for i in range(1000)]


def test_buffer_bounds():
    with Device() as gpu:
        buf = gpu.allocate(4)
        try:
            buf.write(b"12345")
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-bounds write was accepted")
