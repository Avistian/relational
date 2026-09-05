"""Behavioral checks for L049: information flow, live functions, and reference parity."""
import torch
from relkit.claim_models import spa_attention, prompt_weights, prompt_reduce, ExcelFormer


def check():
    torch.set_num_threads(1)
    q = torch.zeros(1, 3, 2, dtype=torch.double)
    v = torch.tensor([[[2., 0.], [4., 0.], [9., 0.]]], dtype=torch.double, requires_grad=True)
    out, a = spa_attention(q, q, v)
    assert torch.allclose(out[0, :, 0], torch.tensor([2., 3., 5.], dtype=torch.double))
    assert torch.equal(a.triu(1), torch.zeros_like(a))
    out[0, 0, 0].backward()
    assert torch.equal(v.grad[0, :, 0], torch.tensor([1., 0., 0.], dtype=torch.double))
    p = torch.tensor([[[1., 0.], [0., 1.]]])
    columns = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    weights = prompt_weights(p, columns)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 2))
    features = torch.arange(12.).reshape(1, 2, 3, 2)
    out = prompt_reduce(weights, features)
    assert out.shape == (1, 2, 2)
    assert torch.allclose(out, torch.stack([(weights[:, j, :, None]*features[:, j]).sum(1) for j in range(2)], 1))
    torch.manual_seed(49)
    model = ExcelFormer(3, d=16, heads=2, layers=2, dropout=0).eval()
    x = torch.randn(5, 3)
    assert torch.allclose(model(x), torch.cat([model(row[None]) for row in x]), atol=1e-6)
    loss = model(x).square().mean(); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return {'checks': 9, 'status': 'PASS'}


if __name__ == '__main__':
    print(check())
