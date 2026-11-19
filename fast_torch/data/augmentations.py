"""Source order: random horizontal flip, REFLECT pad four, crop, normalize."""
import torch
from torch.nn import functional as F

MEAN = (.49139968, .48215841, .44653091)
STD = (.24703223, .24348513, .26158784)

def normalize(images):
    x = images.float()/255.
    return (x-x.new_tensor(MEAN)[None, :, None, None])/x.new_tensor(STD)[None, :, None, None]

def augment(images, flip, rows, columns):
    x = torch.where(flip[:, None, None, None], images.flip(-1), images)
    x = F.pad(x, (4, 4, 4, 4), mode='reflect')
    grid = torch.arange(32, device=x.device)
    x = x.permute(0, 2, 3, 1)[torch.arange(len(x), device=x.device)[:, None, None],
        rows[:, None, None]+grid[None, :, None], columns[:, None, None]+grid[None, None, :]]
    return normalize(x.permute(0, 3, 1, 2))

def epoch_plan(seed, epoch, count=50000):
    # Independent of optimizer, process lifetime, evaluation, compilation and warmup.
    g = torch.Generator(device='cpu').manual_seed(seed+epoch*1000003)
    return (torch.randperm(count, generator=g), torch.rand(count, generator=g) < .5,
            torch.randint(0, 9, (count,), generator=g), torch.randint(0, 9, (count,), generator=g))
