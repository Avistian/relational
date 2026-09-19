"""L076 complete two-table teaching stack. Not the RelBench benchmark model."""
import pandas as pd
import torch
from torch import nn
from torch_frame import stype, NAStrategy
from torch_frame.data import Dataset
from torch_frame.nn import StypeWiseFeatureEncoder, LinearEncoder, EmbeddingEncoder


def fixture():
    customers = pd.DataFrame({'customer_id':[42,7,99,105],
                              'age':[20.,40.,30.,50.],
                              'region':['north','south','north','south']})
    events = pd.DataFrame({'customer_id':[7,42,7,99,42],
                           'amount':[2.,4.,6.,8.,100.],
                           'kind':['buy','buy','return','buy','buy'],
                           'event_day':[1,2,3,4,12],
                           'available_day':[1,2,3,11,12]})
    # Label toy: arbitrary diagnostic targets; never used as features or fitted state.
    return customers, events, torch.tensor([0.,1.,0.,1.])


def key_positions(primary_keys, foreign_keys):
    """Map FK values to positions in this PK ordering; reject broken integrity."""
    if len(set(primary_keys)) != len(primary_keys):
        raise ValueError('Primary keys must be unique')
    lookup = {key:i for i,key in enumerate(primary_keys)}
    if any(key not in lookup for key in foreign_keys):
        raise ValueError('Foreign key has no matching primary key')
    return torch.tensor([lookup[key] for key in foreign_keys], dtype=torch.long)


def eligible_events(event_day, available_day, cutoff):
    """Inclusive availability convention for this lesson's fixed cutoff."""
    return (event_day <= cutoff) & (available_day <= cutoff)


def mean_messages(messages, destination, num_destinations):
    """messages [E,D], destination [E] -> means [N,D], counts [N]."""
    sums = messages.new_zeros((num_destinations, messages.shape[1]))
    sums.index_add_(0, destination, messages)
    counts = torch.bincount(destination, minlength=num_destinations)
    return sums / counts.clamp_min(1).to(messages.dtype).unsqueeze(1), counts


def materialize_tables(customers, events, cutoff=10):
    """PK/FK/time/labels stay outside feature tables. Fit only visible event rows.

    Customer attributes are assumed observed before the cutoff. This fixed-snapshot
    diagnostic has no held-out split. A predictive experiment needs train-only fitting.
    """
    destination = key_positions(customers.customer_id.tolist(), events.customer_id.tolist())
    mask = eligible_events(torch.tensor(events.event_day.to_numpy()),
                           torch.tensor(events.available_day.to_numpy()), cutoff)
    customer_ds = Dataset(customers[['age','region']],
                          col_to_stype={'age':stype.numerical,'region':stype.categorical})
    event_ds = Dataset(events.loc[mask.numpy(), ['amount','kind']].copy(),
                       col_to_stype={'amount':stype.numerical,'kind':stype.categorical})
    customer_ds.materialize(); event_ds.materialize()
    # Conversion of all rows uses frozen visible-row state, not a new fit.
    all_events = event_ds.convert_to_tensor_frame(events[['amount','kind']])
    return customer_ds, event_ds, all_events, destination, mask


class TableEncoder(nn.Module):
    """Real Frame typed tokens [N,C,d] → flattened projection [N,D]."""
    def __init__(self, dataset, width=4):
        super().__init__()
        self.columns = StypeWiseFeatureEncoder(
            out_channels=width, col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
            stype_encoder_dict={stype.numerical:LinearEncoder(na_strategy=NAStrategy.MEAN),
                                stype.categorical:EmbeddingEncoder()})
        self.readout = nn.Sequential(nn.Flatten(1),
                                    nn.Linear(dataset.tensor_frame.num_cols*width,width),nn.Tanh())

    def forward(self, tf):
        tokens, _ = self.columns(tf)
        return self.readout(tokens)


class RelationalStack(nn.Module):
    """One direction, one hop, mean reduction, concatenation and binary head.

    Separate encoders allow different schemas. The head is the update/predictor;
    this is not the two-layer heterogeneous GraphSAGE used by RelBench.
    """
    def __init__(self, customer_ds, event_ds, width=4):
        super().__init__()
        self.customer_encoder = TableEncoder(customer_ds,width)
        self.event_encoder = TableEncoder(event_ds,width)
        self.head = nn.Sequential(nn.Linear(2*width,8),nn.Tanh(),nn.Linear(8,1))

    def forward(self, customer_tf, event_tf, destination, eligible):
        customer_rows = self.customer_encoder(customer_tf)
        event_rows = self.event_encoder(event_tf)
        neighbors, counts = mean_messages(event_rows[eligible],destination[eligible],len(customer_tf))
        combined = torch.cat([customer_rows,neighbors],dim=1)
        logits = self.head(combined).squeeze(-1)
        return logits, {'customers':customer_rows,'events':event_rows,
                        'neighbors':neighbors,'counts':counts,'combined':combined}


def run_diagnostic(steps=100):
    """Overfit four rows: a wiring diagnostic, with NO held-out accuracy claim."""
    torch.set_num_threads(1); torch.manual_seed(76)
    customers, events, labels = fixture()
    cds, eds, etf, destination, mask = materialize_tables(customers,events)
    model = RelationalStack(cds,eds)
    optimizer = torch.optim.Adam(model.parameters(),lr=.03)
    losses=[]
    for step in range(steps):
        optimizer.zero_grad()
        logits, trace = model(cds.tensor_frame,etf,destination,mask)
        loss = nn.functional.binary_cross_entropy_with_logits(logits,labels)
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    with torch.no_grad():logits,trace=model(cds.tensor_frame,etf,destination,mask)
    return {'seed':76,'steps':steps,'first_loss':losses[0],
            'final_loss':float(nn.functional.binary_cross_entropy_with_logits(logits,labels)),
            'losses':losses,'shapes':{k:list(v.shape) for k,v in trace.items()},
            'counts':trace['counts'].tolist(),'logits':logits.tolist()}
