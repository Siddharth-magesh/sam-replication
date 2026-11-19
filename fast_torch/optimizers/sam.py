"""Replica-local SAM, explicit L2 loss, mean -> clip -> Nesterov. No weight mutation."""
import torch
from torch.func import functional_call, grad_and_value, vmap
from torch.nn import functional as F


def norm(tree):
    return torch.sqrt(sum(v.float().square().sum() for v in tree.values()))


def build_gradient(model, rho=.05, wd=.0005, strategy='sequential', autocast=False):
    def loss(parameters, state, images, labels):
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=autocast):
            logits, updated = functional_call(model, parameters, (images, state, True))
        ce = F.cross_entropy(logits.float(), labels)
        return ce+.5*wd*sum(w.float().square().sum() for w in parameters.values()), (updated, ce)
    differentiate = grad_and_value(loss, has_aux=True)
    def lane(parameters, state, images, labels):
        first, (_, (new_state, ce)) = differentiate(parameters, state, images, labels)
        if rho:
            # Exact source formula, no epsilon. Zero gradient is non-finite and must fail validation.
            length = norm(first)
            perturbed = {k: (w+rho*first[k]/length).detach() for k, w in parameters.items()}
            result, (_, (_, ce)) = differentiate(perturbed, state, images, labels)
        else: result = first
        return {k: v.detach() for k, v in result.items()}, new_state, ce.detach()
    def gradients(parameters, state, images, labels):
        replicas = next(iter(state.values())).shape[0]
        images = images.reshape(replicas, -1, *images.shape[1:])
        labels = labels.reshape(replicas, -1)
        if strategy == 'vectorized':
            gs, states, losses = vmap(lane, in_dims=(None, 0, 0, 0))(parameters, state, images, labels)
            return {k: v.mean(0) for k, v in gs.items()}, states, losses.mean()
        if strategy != 'sequential': raise ValueError(strategy)
        averaged = {k: torch.zeros_like(v) for k, v in parameters.items()}
        states = {k: [] for k in state}; losses = []
        for i in range(replicas):
            g, s, ce = lane(parameters, {k: v[i] for k, v in state.items()}, images[i], labels[i])
            for k in g: averaged[k] = averaged[k]+g[k]/replicas
            for k in s: states[k].append(s[k])
            losses.append(ce)
        return averaged, {k: torch.stack(v) for k, v in states.items()}, torch.stack(losses).mean()
    return gradients


def clip(gradients, threshold=5.):
    length = norm(gradients)
    return {k: torch.where(length < threshold, v, (v/length)*threshold) for k, v in gradients.items()}


class Stepper:
    def __init__(self, model, rho=.05, strategy='sequential', compile=False, optimizer='plain', bf16=False):
        self.model = model
        self.gradient = build_gradient(model, rho=rho, strategy=strategy, autocast=bf16)
        if compile:
            # Fail loudly on graph breaks; never silently advertise an eager fallback as compiled.
            self.gradient = torch.compile(self.gradient, fullgraph=True, dynamic=False)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, nesterov=True,
            weight_decay=0., foreach=optimizer == 'foreach', fused=optimizer == 'fused')

    def step(self, state, images, labels, lr):
        parameters = dict(self.model.named_parameters())
        gradients, new_state, loss = self.gradient(parameters, state, images, labels)
        gradients = clip(gradients)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm(gradients))):
            raise FloatingPointError('Non-finite loss/gradient (including zero-norm SAM case)')
        for k, p in parameters.items(): p.grad = gradients[k].detach()
        for group in self.optimizer.param_groups: group['lr'] = lr
        self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        return {k: v.detach() for k, v in new_state.items()}, loss.detach()
