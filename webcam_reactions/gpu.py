"""
GPU (OpenCL) helpers for image compositing.

OpenCV's transparent API (cv2.UMat) runs remap / resize / blending on the GPU through OpenCL
(NVIDIA, AMD and Intel drivers all provide it on Windows). Everything falls back to the CPU
(numpy) path when OpenCL is missing or USE_OPENCL is False.
"""
import cv2
import numpy as np

from . import config as C

_STATE = {"checked": False, "on": False}


def enabled():
    if not _STATE["checked"]:
        _STATE["checked"] = True
        try:
            if C.USE_OPENCL and cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                _STATE["on"] = cv2.ocl.useOpenCL()
        except Exception:
            _STATE["on"] = False
    return _STATE["on"]


def warmup(width, height):
    """Compile the OpenCL kernels now (first use takes ~0.2 s), not in the middle of an effect."""
    if not enabled():
        return
    from .flag import WavingFlag
    bg = WavingFlag(width, height).frame_gpu(0.0)
    frame = cv2.UMat(np.zeros((height, width, 3), np.uint8))
    m = cv2.UMat(np.zeros((height, width), np.float32))
    blend_into(frame, bg, cv2.addWeighted(m, -1.0, m, 0.0, 1.0))
    top, a = upload_bgra(np.zeros((64, 64, 4), np.uint8))
    blend_into(roi(frame, 0, 0, 64, 64), top, a)
    frame.get()


def device_name():
    try:
        return cv2.ocl.Device.getDefault().name() if enabled() else "CPU"
    except Exception:
        return "CPU"


def upload_bgra(img, opacity=1.0):
    """BGRA numpy image -> (BGR UMat, float alpha UMat scaled by opacity); one upload, split on the GPU."""
    b, g, r, a8 = cv2.split(cv2.UMat(np.ascontiguousarray(img)))
    return cv2.merge([b, g, r]), cv2.multiply(a8, opacity / 255.0, dtype=cv2.CV_32F)


def roi(umat, x0, y0, x1, y1):
    """Writable view of a rectangle of a UMat."""
    return cv2.UMat(umat, (int(y0), int(y1)), (int(x0), int(x1)))


def blend_into(dst, top, alpha):
    """dst = dst * (1 - alpha) + top * alpha, in place. dst may be a numpy array or a UMat (ROI);
    top / alpha may be numpy (uploaded here) or UMat. alpha: float32 2-D map (0-1)."""
    if isinstance(dst, cv2.UMat):
        a = alpha if isinstance(alpha, cv2.UMat) else cv2.UMat(np.ascontiguousarray(alpha, np.float32))
        t = top if isinstance(top, cv2.UMat) else cv2.UMat(np.ascontiguousarray(top))
        inv = cv2.addWeighted(a, -1.0, a, 0.0, 1.0)              # 1 - alpha, computed on the GPU
        cv2.blendLinear(dst, t, inv, a, dst=dst)
    else:
        alpha = np.ascontiguousarray(alpha, np.float32)
        cv2.blendLinear(dst, np.ascontiguousarray(top), 1.0 - alpha, alpha, dst=dst)
    return dst
