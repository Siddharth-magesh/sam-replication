"""Independent NCHW rendition of sam_jax/models/wide_resnet.py (not authors' code)."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def specification(blocks=4, width=10, classes=10):
    """Canonical source paths -> PyTorch shapes; shortcuts contain no parameters."""
    shapes = {'init_conv/kernel': (16, 3, 3, 3)}
    bn = {}; incoming = 16
    for group, channels in enumerate((16*width, 32*width, 64*width)):
        for block in range(blocks):
            root = f'WideResnetGroup_{group}/WideResnetBlock_{block}'
            for name, size in [('init_bn', incoming), ('bn_2', channels)]:
                bn[root+'/'+name] = size
                shapes[root+'/'+name+'/scale'] = (size,)
                shapes[root+'/'+name+'/bias'] = (size,)
            shapes[root+'/conv1/kernel'] = (channels, incoming, 3, 3)
            shapes[root+'/conv2/kernel'] = (channels, channels, 3, 3)
            incoming = channels
    bn['pre-pool-bn'] = incoming
    shapes['pre-pool-bn/scale'] = (incoming,)
    shapes['pre-pool-bn/bias'] = (incoming,)
    shapes['Dense_0/kernel'] = (classes, incoming)
    shapes['Dense_0/bias'] = (classes,)
    return shapes, bn


def key(path):
    return path.replace('/', '__').replace('-', '_')


class WideResNet(nn.Module):
    def __init__(self, blocks=4, width=10, classes=10):
        super().__init__()
        self.blocks, self.width, self.classes = blocks, width, classes
        self.shapes, self.bn_shapes = specification(blocks, width, classes)
        self.p = nn.ParameterDict()
        for name, shape in self.shapes.items():
            value = torch.empty(shape)
            if len(shape) == 4:
                # JAX variance_scaling(2, fan_out, normal): untruncated Gaussian.
                nn.init.normal_(value, std=math.sqrt(2/(shape[0]*shape[2]*shape[3])))
            elif name.endswith('/kernel'):
                nn.init.uniform_(value, -1/math.sqrt(classes), 1/math.sqrt(classes))
            elif name.endswith('/scale'): nn.init.ones_(value)
            else: nn.init.zeros_(value)
            self.p[key(name)] = nn.Parameter(value)

    def initial_bn(self, lanes=None):
        sample = next(self.parameters())
        result = {}
        for name, size in self.bn_shapes.items():
            shape = (size,) if lanes is None else (lanes, size)
            result[name+'/mean'] = sample.new_zeros(shape)
            result[name+'/var'] = sample.new_ones(shape)
        return result

    def forward(self, x, state, training=True):
        new_state = {}
        def activation(x, name):
            # Flax 0.2.2 uses E[x^2]-E[x]^2, not torch BatchNorm's running_var correction.
            x = x.float()
            if training:
                mean = x.mean((0, 2, 3)); var = x.square().mean((0, 2, 3))-mean.square()
                new_state[name+'/mean'] = (.9*state[name+'/mean']+.1*mean).detach()
                new_state[name+'/var'] = (.9*state[name+'/var']+.1*var).detach()
            else:
                mean, var = state[name+'/mean'], state[name+'/var']
                new_state[name+'/mean'], new_state[name+'/var'] = mean, var
            scale = self.p[key(name+'/scale')]
            bias = self.p[key(name+'/bias')]
            multiplier = torch.rsqrt(var+1e-5)*scale
            return F.relu((x-mean[None, :, None, None])*multiplier[None, :, None, None]+bias[None, :, None, None])
        def conv(x, name, stride=1):
            # TensorFlow SAME: stride-2 even input has pad 0 before, 1 after.
            pads = []
            for size in reversed(x.shape[-2:]):
                total = max(0, ((size+stride-1)//stride-1)*stride+3-size)
                pads.extend([total//2, total-total//2])
            return F.conv2d(F.pad(x, pads), self.p[key(name+'/kernel')], stride=stride)
        x = conv(x, 'init_conv')
        for group, channels in enumerate((16*self.width, 32*self.width, 64*self.width)):
            for block in range(self.blocks):
                name = f'WideResnetGroup_{group}/WideResnetBlock_{block}'
                residual = x
                x = activation(x, name+'/init_bn')
                if group == 0 and block == 0: residual = x
                stride = 2 if group > 0 and block == 0 else 1
                x = conv(x, name+'/conv1', stride)
                x = conv(activation(x, name+'/bn_2'), name+'/conv2')
                if residual.shape[1] != channels:
                    residual = F.avg_pool2d(residual, stride, stride)
                    residual = F.pad(residual, (0, 0, 0, 0, 0, channels-residual.shape[1]))
                x = x+residual
        x = activation(x, 'pre-pool-bn').mean((2, 3))
        return F.linear(x, self.p[key('Dense_0/kernel')], self.p[key('Dense_0/bias')]), new_state

    def expected_parameters(self):
        return sum(math.prod(shape) for shape in self.shapes.values())
