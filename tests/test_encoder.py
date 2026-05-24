import torch


from src.models.encoder import Encoder


def test_encoder_smoke():
    torch.manual_seed(0)
    model = Encoder(
        in_channels=3,
        num_hiddens=128,
        num_residual_blocks=2,
        num_residual_hiddens=32,
    )

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = model(x)

    assert y.shape == (2, 128, 16, 16)
    y.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters())
