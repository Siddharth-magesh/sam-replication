"""CPU only; no dataset download or CUDA initialization."""
import json
from fast_torch.validation.parity_tests import run
if __name__ == '__main__': print(json.dumps(run(), indent=2))
