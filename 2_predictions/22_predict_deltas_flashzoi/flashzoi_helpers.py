# flashzoi_helpers.py
from typing import List
import torch
from borzoi_pytorch import Borzoi   # pip install borzoi-pytorch

def load_flashzoi_models(num_folds: int,
                         device: torch.device,
                         use_autocast: bool = True) -> List[Borzoi]:
    """
    Download and prepare multiple Flashzoi folds for ensemble scoring.

    Parameters
    ----------
    num_folds : int           # e.g. 4
    device     : torch.device # cuda / mps / cpu
    use_autocast: bool        # mixed precision if on GPU

    Returns
    -------
    List[Borzoi]  (eval()-mode, on device)
    """
    models: List[Borzoi] = []
    dtype_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and use_autocast
        else torch.autocast(enabled=False, device_type=device.type)
    )

    print(f"Loading {num_folds} Flashzoi model folds…")
    with dtype_ctx:
        for f in range(num_folds):
            name   = f"johahi/flashzoi-replicate-{f}"
            model  = Borzoi.from_pretrained(name).to(device).eval()
            try:
                model = torch.compile(model)      # PyTorch ≥2.0, safe no-op otherwise
            except Exception:
                pass
            models.append(model)
            print(f"  • fold {f} loaded")

    return models