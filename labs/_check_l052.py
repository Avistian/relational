"""Behavioral checks: identity exclusion, value direction, gradients and context integrity."""
import json
import torch
from relkit.tabr import TabRS, eligible_mask, select_neighbors, context_value


def check():
    torch.set_num_threads(1)
    torch.manual_seed(52)
    qids = torch.tensor([9, 3])
    cids = torch.tensor([3, 7, 9])
    mask = eligible_mask(qids, cids)
    assert mask.tolist() == [[True, True, False], [False, True, True]]
    # A duplicate feature vector with a DIFFERENT identity remains eligible.
    q = torch.tensor([[0., 0.], [2., 0.]], requires_grad=True)
    c = torch.tensor([[2., 0.], [0., 0.], [0., 0.]], requires_grad=True)
    idx = select_neighbors(q, c, 1, mask)
    assert idx.tolist() == [[1], [1]]
    try:
        select_neighbors(q, c, 3, mask)
    except ValueError:
        pass
    else:
        raise AssertionError('Too few legal neighbors must fail explicitly')
    value = context_value(q, c[idx], torch.ones(2, 1, 2), torch.nn.Identity())
    assert torch.equal(value, 1 + q[:, None] - c[idx])
    value.sum().backward()
    assert q.grad.abs().sum() > 0 and c.grad.abs().sum() > 0
    x = torch.randn(9, 3)
    y = torch.arange(9) % 2
    model = TabRS(3, d=8, m=3, dropout=0, context_dropout=0)
    model.eval()
    ids = torch.arange(9)
    a = model(x[:2], x, y, ids[:2], ids)
    b = torch.cat([model(x[i:i+1], x, y, ids[i:i+1], ids) for i in range(2)])
    torch.testing.assert_close(a, b)
    permutation = torch.tensor([8, 2, 6, 0, 4, 1, 7, 3, 5])
    torch.testing.assert_close(a, model(x[:2], x[permutation], y[permutation], ids[:2], ids[permutation]))
    changed = y.clone(); changed[0] = 1 - changed[0]
    torch.testing.assert_close(a[:1], model(x[:1], x, changed, ids[:1], ids))
    model.train()
    model(x[:2], x, y, ids[:2], ids).sum().backward()
    assert model.K.weight.grad.abs().sum() > 0
    assert model.T[0].weight.grad.abs().sum() > 0
    assert model.label_encoder.weight.grad.abs().sum() > 0
    regression = TabRS(3, d=8, m=3, regression=True)
    assert regression(x[:2], x, y.float(), ids[:2], ids).shape == (2,)
    return {'status': 'PASS', 'checks': 11}


if __name__ == '__main__':
    print(json.dumps(check()))
