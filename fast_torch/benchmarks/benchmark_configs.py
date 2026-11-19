"""Candidates only; no speed claims or automatic promotion to production."""
BASE = dict(profile='fp32_reference', strategy='sequential', compile=False,
            channels_last=False, optimizer='plain', gpu_data=False, benchmark=False, deterministic=True)
CASES = {'reference': BASE}
for name, changes in {
    'vectorized': dict(strategy='vectorized'),
    'compile': dict(compile=True),
    'channels_last': dict(channels_last=True),
    'foreach': dict(optimizer='foreach'),
    'fused': dict(optimizer='fused'),
    'gpu_data': dict(gpu_data=True),
    'cudnn_benchmark': dict(benchmark=True, deterministic=False),
}.items():
    CASES[name] = dict(BASE, profile='fp32_optimized', **changes)
CASES['tf32'] = dict(BASE, profile='tf32_optimized')
CASES['bf16'] = dict(BASE, profile='bf16_optimized')

def configure(case):
    import torch
    tf32 = case['profile'] == 'tf32_optimized'
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = case['benchmark']
    torch.backends.cudnn.deterministic = case['deterministic']
    torch.use_deterministic_algorithms(case['deterministic'])
