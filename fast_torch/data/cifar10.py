"""Load an existing official CIFAR binary archive. Never downloads implicitly."""
import hashlib
import tarfile
import numpy as np
import torch
from .augmentations import augment, normalize, epoch_plan

ARCHIVE_SHA256 = 'c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd'

def load(archive, device='cpu'):
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA256, 'CIFAR archive hash mismatch'
    splits = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for split, names in [('train', [f'data_batch_{i}.bin' for i in range(1, 6)]), ('test', ['test_batch.bin'])]:
            batches = []
            for name in names:
                members = [m for m in tar.getmembers() if m.name.endswith('/'+name)]
                assert len(members) == 1 and members[0].size == 10000*3073
                batches.append(np.frombuffer(tar.extractfile(members[0]).read(), dtype=np.uint8).reshape(-1, 3073))
            data = np.concatenate(batches)
            splits[split] = (torch.from_numpy(data[:, 1:].copy().reshape(-1, 3, 32, 32)).to(device),
                             torch.from_numpy(data[:, 0].astype(np.int64)).to(device))
    return splits

def train_batches(data, seed, epoch, device):
    images, labels = data
    permutation, flips, rows, columns = epoch_plan(seed, epoch, len(images))
    for start in range(0, len(images)-255, 256):
        idx = permutation[start:start+256].to(images.device)
        args = [x[start:start+256].to(device) for x in (flips, rows, columns)]
        yield augment(images[idx].to(device), *args), labels[idx].to(device)

def eval_batches(data, device):
    images, labels = data
    for start in range(0, len(images), 256):
        yield normalize(images[start:start+256].to(device)), labels[start:start+256].to(device)
