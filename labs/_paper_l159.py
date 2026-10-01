"""Fail-closed audit command, deliberately NOT a benchmark trainer."""
import json
if __name__=='__main__':
    print(json.dumps(dict(target='Vogel et al. 2023 Table1 wikiTables row baseline versus GNN; three tasks, three runs',
        status='NOT_RUN',fidelity='NOT_ESTABLISHED',runnable_historical_trainer=False,
        blockers=['exact corpus subset and splits','released implementation and checkpoints','model and training configuration','masking and text accuracy conventions'],
        local_mechanism='SEPARATE_INCOMPARABLE_EXPERIMENT',cloud_spend_usd=0),indent=2))
    raise SystemExit(2)
