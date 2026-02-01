"""
device_manager.py
─────────────────────────────────────────────────────────────────────
Abstracts all hardware-selection logic in one place.
    from device_manager import DeviceManager
    dm = DeviceManager(cfg.device)
    device = dm.device          # torch.device
    dtype  = dm.get_torch_dtype("auto")
─────────────────────────────────────────────────────────────────────
"""

import torch

from config import DeviceConfig
from logger import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """Resolve the best available torch device and associated dtype."""

    def __init__(self, cfg: DeviceConfig):
        self._cfg = cfg
        self._device = self._resolve_device(cfg.device)
        logger.info("Selected device: %s", self._device)

    # ── public ───────────────────────────────────────────────────
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_cuda(self) -> bool:
        return self._device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        return self._device.type == "mps"

    @property
    def is_cpu(self) -> bool:
        return self._device.type == "cpu"

    def get_torch_dtype(self, dtype_str: str = "auto") -> torch.dtype:
        """
        Map a string dtype token to a torch.dtype.
        "auto" picks float16 on CUDA / bfloat16 on MPS / float32 on CPU.
        """
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        if dtype_str != "auto":
            if dtype_str not in mapping:
                raise ValueError(
                    f"Unsupported dtype '{dtype_str}'. "
                    f"Choose from: {list(mapping.keys())} or 'auto'."
                )
            return mapping[dtype_str]

        # ── auto logic ───────────────────────────────────────────
        if self.is_cuda:
            chosen = torch.float16
        elif self.is_mps:
            chosen = torch.bfloat16
        else:
            chosen = torch.float32

        logger.info("Auto-selected dtype: %s", chosen)
        return chosen

    def gpu_memory_info(self) -> dict:
        """Return a dict with allocated / reserved / total GPU memory (GiB)."""
        if not self.is_cuda:
            return {}
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        return {"allocated_gib": round(alloc, 2),
                "reserved_gib": round(reserved, 2),
                "total_gib": round(total, 2)}

    # ── private ──────────────────────────────────────────────────
    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        """Pick the best device based on availability and user preference."""
        pref = preference.lower().strip()

        if pref == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if pref in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("cuda")

        if pref == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("mps")

        if pref == "cpu":
            return torch.device("cpu")

        raise ValueError(
            f"Unknown device preference '{preference}'. "
            "Use 'auto', 'cuda', 'mps', or 'cpu'."
        )
