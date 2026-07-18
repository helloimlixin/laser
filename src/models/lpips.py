from collections import namedtuple
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import vgg16

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

REPO_ROOT = Path(__file__).resolve().parents[2]
RQVAE_CKPT_ROOT = REPO_ROOT / "rqvae" / "losses" / "vqgan" / ".caches"
BUNDLED_CKPT_ROOT = REPO_ROOT / "vgg_lpips"
VGG16_CKPT_CANDIDATES = (
    "vgg16_features.pth",
    "vgg16-397923af.pth",
    "vgg16.pth",
)


def _torch_load(path):
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def _checkpoint_state_dict(path):
    checkpoint = _torch_load(path)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Checkpoint at {path} did not contain a state dict")


def get_ckpt_path(name):
    if name not in CKPT_MAP:
        raise ValueError(f"Unknown LPIPS checkpoint {name!r}")
    for root in (RQVAE_CKPT_ROOT, BUNDLED_CKPT_ROOT):
        path = root / CKPT_MAP[name]
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"LPIPS checkpoint not found. Expected RQ-VAE-style checkpoint at "
        f"{RQVAE_CKPT_ROOT / CKPT_MAP[name]} or compatibility fallback at "
        f"{BUNDLED_CKPT_ROOT / CKPT_MAP[name]}."
    )


def _candidate_vgg16_weight_paths():
    explicit = str(os.environ.get("LASER_VGG16_WEIGHTS", "") or "").strip()
    if explicit:
        yield Path(explicit).expanduser()
    try:
        hub_dir = Path(torch.hub.get_dir()).expanduser() / "checkpoints"
        yield hub_dir / "vgg16-397923af.pth"
    except Exception:
        pass
    for filename in VGG16_CKPT_CANDIDATES:
        yield BUNDLED_CKPT_ROOT / filename


def get_vgg16_weights_path():
    for candidate in _candidate_vgg16_weight_paths():
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(path) for path in _candidate_vgg16_weight_paths())
    raise FileNotFoundError(
        "Local VGG16 feature weights not found. LPIPS is offline-only here: "
        "vgg_lpips/vgg.pth contains the LPIPS linear calibration layers, "
        "but VGG16 convolution weights are also required. Place vgg16-397923af.pth "
        "or vgg16_features.pth under vgg_lpips/, or set LASER_VGG16_WEIGHTS. "
        f"No online download was attempted. Tried: {tried}"
    )


def _normalize_key(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _vgg_features_state_dict(checkpoint_path, features: nn.Module):
    state_dict = _checkpoint_state_dict(checkpoint_path)
    feature_keys = set(features.state_dict().keys())
    out = {}
    for raw_key, value in state_dict.items():
        key = _normalize_key(str(raw_key))
        if key.startswith("features."):
            key = key[len("features."):]
        if key in feature_keys:
            out[key] = value
    missing = sorted(feature_keys.difference(out))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise RuntimeError(
            f"VGG16 checkpoint {checkpoint_path} is missing feature weights: {preview}{suffix}"
        )
    return out


class LPIPS(nn.Module):
    def __init__(self, version="0.1"):
        super(LPIPS, self).__init__()
        self.version = str(version).strip().lower().removeprefix("v")
        if self.version not in {"0.0", "0.1"}:
            raise ValueError(f"LPIPS version must be '0.0' or '0.1', got {version!r}")
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.vgg = VGG16()

        self.lin0 = NetLinLayer(self.channels[0])
        self.lin1 = NetLinLayer(self.channels[1])
        self.lin2 = NetLinLayer(self.channels[2])
        self.lin3 = NetLinLayer(self.channels[3])
        self.lin4 = NetLinLayer(self.channels[4])
        self.linear_layers = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])

        self.load_from_pretrained()

        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def train(self, mode: bool = True):
        """Keep the frozen perceptual network in inference mode.

        LPIPS contains dropout in its learned linear calibration heads.  Because
        this module is registered under LASER, ``LASER.train()`` would otherwise
        recursively re-enable that dropout even though LPIPS is a frozen loss
        network.  RQ-VAE explicitly keeps its LPIPS instance in eval mode.
        """
        return super().train(False)

    def _prepare_input(self, x):
        # Official LPIPS v0.0 omitted this scaling step; v0.1 added it.  Keep
        # the learned linear calibration layers in both modes so this switch
        # changes only the historical normalization behavior.
        return self.scaling_layer(x) if self.version == "0.1" else x

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name)
        state_dict = _checkpoint_state_dict(ckpt)
        incompatible = self.load_state_dict(state_dict, strict=False)
        print(f"loaded pretrained LPIPS loss from {ckpt}")
        expected = {f"lin{i}.model.1.weight" for i in range(5)}
        missing = sorted(expected.intersection(incompatible.missing_keys))
        if missing:
            raise RuntimeError(f"LPIPS checkpoint {ckpt} is missing weights: {missing}")

    def forward(self, real_x, fake_x):
        outs_real = self.vgg(self._prepare_input(real_x))
        outs_fake = self.vgg(self._prepare_input(fake_x))
        features_real, features_fake, diffs = {}, {}, {}

        for i in range(len(self.channels)):
            features_real[i], features_fake[i] = norm_tensor(outs_real[i]), norm_tensor(outs_fake[i])
            diffs[i] = (features_real[i] - features_fake[i]) ** 2

        return sum([spatial_average(self.linear_layers[i](diffs[i])) for i in range(len(self.channels))])


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    def __init__(self, weights_path=None):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(weights=None).features
        weights_path = Path(weights_path).expanduser().resolve() if weights_path is not None else get_vgg16_weights_path()
        vgg_pretrained_features.load_state_dict(
            _vgg_features_state_dict(weights_path, vgg_pretrained_features),
            strict=True,
        )
        self.weights_path = str(weights_path)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VGGOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def norm_tensor(x):
    """
    Normalize images by their length to make them unit vector?
    :param x: batch of images
    :return: normalized batch of images
    """
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + 1e-10)
    return x / norm_factor


def spatial_average(x):
    """
     imgs have: batch_size z_e channels z_e width z_e height --> average over width and height channel
    :param x: batch of images
    :return: averaged images along width and height
    """
    return x.mean([2, 3], keepdim=True)
