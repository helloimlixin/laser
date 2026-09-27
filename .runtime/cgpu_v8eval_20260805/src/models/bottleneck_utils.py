from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SparseCodes:
    support: torch.Tensor
    values: torch.Tensor
    num_embeddings: int
    code_format: str = "flat"


def _normalize_dictionary(dictionary: torch.Tensor, eps: float) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(dictionary), p=2, dim=0, eps=eps)
