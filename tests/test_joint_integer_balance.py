import torch

from src.training.joint_integer_balance import joint_integer_balance


def test_joint_balance_diagnostics_match_independent_autograd_components():
    torch.manual_seed(421)
    features = torch.randn(2, 2, 4, 5, requires_grad=True)
    weight = torch.randn(21, 5)
    logits = features@weight.T
    atoms = torch.randint(7, (2, 2, 4))
    target = torch.randn(2, 2, 4, 3).softmax(-1)
    report = joint_integer_balance(logits, atoms, target, classifier_weight=weight, row_chunk=3)
    ids = atoms[..., None]*3+torch.arange(3)
    selected = logits.gather(-1, ids)
    atom_loss = -logits.softmax(-1).reshape(2, 2, 4, 7, 3).sum(-1).gather(-1, atoms[..., None]).squeeze(-1).log()
    coefficient_loss = -(target*selected.log_softmax(-1)).sum(-1)
    gradients = {}
    for name, loss in [('atom', atom_loss), ('coefficient', coefficient_loss), ('joint', atom_loss+coefficient_loss)]:
        gradients[name] = torch.autograd.grad(loss.sum(), (logits, features), retain_graph=True)
    values = torch.tensor(report['sums'])/torch.tensor(report['counts'])[:, None]
    for d in range(4):
        expected = [float((atom_loss+coefficient_loss)[..., d].mean()),
                    float(atom_loss[..., d].mean()), float(coefficient_loss[..., d].mean())]
        torch.testing.assert_close(values[d, :3], torch.tensor(expected), rtol=2.e-5, atol=2.e-6)
        for index, name in [(4, 'atom'), (5, 'coefficient'), (6, 'joint')]:
            torch.testing.assert_close(values[d, index], gradients[name][0][..., d, :].square().sum(-1).mean(), rtol=2.e-5, atol=2.e-6)
        for index, name in [(7, 'atom'), (8, 'coefficient'), (9, 'joint')]:
            torch.testing.assert_close(values[d, index], gradients[name][1][..., d, :].square().sum(-1).mean(), rtol=2.e-5, atol=2.e-6)
    assert report['objective_weights'] == dict(atom_marginal=1., coefficient_conditional=1., depth=[.25]*4)
