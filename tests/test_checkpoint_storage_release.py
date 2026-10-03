import gc
import weakref

import torch

from src.training.rqtransformer import atomic_torch_save


def test_saved_checkpoint_does_not_pin_replaced_optimizer_storage(tmp_path):
    """Offloading Adam state after a save must release its original storage."""
    was_enabled = gc.isenabled()
    gc.disable()  # Expose cycles deterministically rather than waiting for GC.
    try:
        moment = torch.arange(64, dtype=torch.float32)
        storage = weakref.ref(moment.untyped_storage())
        state = {"exp_avg": moment}
        atomic_torch_save({"optimizer": state}, tmp_path / "last.pt")
        state["exp_avg"] = moment.clone()
        del moment
        assert storage() is None
        restored = torch.load(tmp_path / "last.pt", weights_only=True)
        torch.testing.assert_close(restored["optimizer"]["exp_avg"], state["exp_avg"])
    finally:
        if was_enabled:
            gc.enable()
