"""Minimal model interfaces used by the vendored RQ-VAE implementations."""

import abc

from torch import nn


class Stage1Model(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_codes(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def decode_code(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_recon_imgs(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError


class Stage2Model(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size
