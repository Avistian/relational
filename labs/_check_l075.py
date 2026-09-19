"""Behavioral contracts: five stypes, leakage, schema alignment and live TODOs."""
import copy
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from relkit.frame_l075 import fixture, schema, fit_materializer, encoder_config, RowEncoder, numeric_tokens

def check():
    torch.manual_seed(75)
    train, query = fixture()
    ds, held = fit_materializer(train, query, schema())
    assert ds.tensor_frame.num_cols == 5 and held.num_cols == 5
    assert ds.col_stats['amount'][StatType.MEAN] == 20
    assert held.feat_dict[stype.categorical][0, 0].item() == -1
    assert held.feat_dict[stype.categorical][1, 0].item() == -1
    altered = query.copy(deep=True); altered['amount'] = [1e8, -1e8]
    ds2, _ = fit_materializer(train, altered, schema())
    assert ds2.col_stats['amount'][StatType.MEAN] == 20
    model = RowEncoder(ds, width=8, row_width=6)
    tok, names = model.columns(held)
    assert tok.shape == (2, 5, 8) and set(names) == set(schema())
    assert torch.isfinite(tok).all() and model(held).shape == (2, 6)
    categorical = names.index('region')
    assert torch.equal(tok[:, categorical], torch.zeros(2, 8))
    # Changing DataFrame column order must not silently reassign learned roles.
    _, reordered = fit_materializer(train, query[list(reversed(query.columns))], schema())
    assert torch.allclose(model(held), model(reordered))
    # A row encoder is row-local: neighbor rows do not alter this output.
    assert torch.allclose(model(held)[:1], model(held[:1]), atol=1e-6)
    model(held).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    enc = model.columns.encoder_dict['numerical']
    raw = ds.tensor_frame.feat_dict[stype.numerical]
    expected = enc(raw)
    actual = numeric_tokens(raw, enc.mean, enc.std, enc.weight, enc.bias)
    assert torch.allclose(actual, expected, atol=1e-7)
    # Independent hand oracle: z=(30-20)/10=1; [2,-1]*1+[.5,.5].
    manual = numeric_tokens(torch.tensor([[30.]]), torch.tensor([20.]), torch.tensor([10.]), torch.tensor([[2.,-1.]]), torch.tensor([[.5,.5]]))
    assert torch.equal(manual, torch.tensor([[[2.5,-.5]]]))
    missing = numeric_tokens(torch.tensor([[float('nan')]]), torch.tensor([20.]), torch.tensor([10.]), torch.ones(1,2), torch.ones(1,2))
    assert torch.equal(missing, torch.ones(1,1,2))
    return {'status':'PASS','tokens':list(tok.shape),'rows':list(model(held).shape),'column_order':names,'train_mean':20.0,'unseen_and_missing_id':-1,'numeric_parity_max_error':float((actual-expected).abs().max().detach()),'gradient_flow':True,'row_locality':True}

if __name__ == '__main__':
    print(check())
