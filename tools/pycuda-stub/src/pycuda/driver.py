"""Stub pycuda.driver - raises RuntimeError on any actual use."""


def init():
    raise RuntimeError("pycuda stub: no CUDA available")


class Device:
    @staticmethod
    def count():
        return 0

    def __init__(self, *args, **kwargs):
        raise RuntimeError("pycuda stub: no CUDA available")
