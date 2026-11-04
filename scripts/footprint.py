from __future__ import annotations

import ctypes
import os
import resource

_RUSAGE_INFO_V4 = 4
_LIFETIME_MAX_FOOTPRINT_OFFSET = 240


def peak_footprint_gb() -> float:
    buffer = ctypes.create_string_buffer(512)
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    if libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, buffer) != 0:
        raise OSError("proc_pid_rusage failed")
    return int.from_bytes(buffer.raw[_LIFETIME_MAX_FOOTPRINT_OFFSET : _LIFETIME_MAX_FOOTPRINT_OFFSET + 8], "little") / 1e9


def peak_resident_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
