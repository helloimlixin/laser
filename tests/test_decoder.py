import torch


from src.models.decoder import Decoder, SimpleDecoder


def test_decoder_smoke():
    torch.manual_seed(0)
    model = Decoder(
        ch=32,
        out_ch=3,
        ch_mult=(1, 2, 4),
        num_res_blocks=1,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=64,
        z_channels=32,
        use_mid_attention=False,
        extra_res_blocks=0,
    )

    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    y = model(x)

    assert y.shape == (2, 3, 64, 64)
    y.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters())


def test_simple_decoder_smoke():
    torch.manual_seed(0)
    model = SimpleDecoder(
        in_channels=128,
        num_hiddens=128,
        num_residual_blocks=2,
        num_residual_hiddens=32,
        out_channels=3,
        num_upsamples=2,
    )

    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    y = model(x)

    assert y.shape == (2, 3, 64, 64)
    y.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters())
