"""Expose a released local-normalization rounding edge case, without a model."""
import hashlib, importlib.util, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
source = ROOT / 'data/cache/l067-source/official/pfn.py'
manifest = json.loads((ROOT / '_sources_l067_v2.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files']['pfn.py']['sha256']
spec = importlib.util.spec_from_file_location('l067_original_normalization', source)
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
torch.set_num_threads(1)
value = -1.3079352378845215
x = torch.zeros(33, 4, 100, dtype=torch.float32)
x[:, :, 3] = value
single = original.normalize_data(x, 32) * (100/8)
double = original.normalize_data(x.double(), 32) * (100/8)
raw = x.transpose(0, 1)[..., :8].contiguous()
shortcut = ((raw-raw[:, :32].mean(1, keepdim=True)) / (raw[:, :32].std(1, keepdim=True)+1e-6)).clamp(-100, 100)*(100/8)
assert x[:32, :, 3].unique().numel() == 1
assert double.abs().max() == 0
assert abs(float(single[0, 0, 3])) > 1
report = dict(status='PASS', source_sha256=manifest['files']['pfn.py']['sha256'],
              shape=list(x.shape), context_rows=32, active_features=8,
              constant_float32_value=value, original_float32=float(single[0, 0, 3]),
              original_float64=float(double[0, 0, 3]),
              mean_std_shortcut_float32=float(shortcut[0, 0, 3]), torch_version=torch.__version__,
              scope='Original explicit masked sum/count and squared deviations on constant context features; finite-precision residual amplified by epsilon and 100/F. Numerical implementation fidelity, not a new learned-model experiment.')
(ROOT / '_normalization_l067_results.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
