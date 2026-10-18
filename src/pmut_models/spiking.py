import torch
from torch import nn


class SpikeFunction(torch.autograd.Function):
    """Heaviside spike with fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.slope = slope
        return (input_tensor > 0.0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (slope * torch.abs(input_tensor) + 1.0) ** 2
        return grad, None


def spike(input_tensor: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    return SpikeFunction.apply(input_tensor, slope)


class LIFStateLayer(nn.Module):
    def __init__(self, beta: float = 0.9, threshold: float = 1.0, surrogate_slope: float = 25.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope

    def forward(self, input_current: torch.Tensor, membrane: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if membrane is None:
            membrane = torch.zeros_like(input_current)
        membrane = self.beta * membrane + input_current
        spikes = spike(membrane - self.threshold, self.surrogate_slope)
        membrane = membrane - spikes.detach() * self.threshold
        return spikes, membrane


class SpikeActivationHook:
    def __init__(self, name: str):
        self.name = name
        self.activations = []

    def __call__(self, module, input_tensor, output):
        self.activations.append(output.detach().cpu())

    def get_activation_map(self, clear: bool = True):
        if not self.activations:
            return None
        result = torch.cat(self.activations, dim=0)
        if clear:
            self.activations.clear()
        return result






class SpikingCNN1D(nn.Module):
    """1D convolution over sweep axis followed by LIF spiking blocks with BatchNorm."""

    def __init__(self, input_shape: tuple[int, int], num_classes: int, hidden_channels: int = 32):
        super().__init__()
        _, channels = input_shape
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels * 2),
        )
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.lif1 = LIFStateLayer(beta=0.9)
        self.lif2 = LIFStateLayer(beta=0.9)
        self.head = nn.Linear(hidden_channels * 2, num_classes)
        self._hooks: list[SpikeActivationHook] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        z1 = self.conv1(x)
        spk1, _ = self.lif1(z1)
        z2 = self.pool(spk1)
        z2 = self.conv2(z2)
        spk2, _ = self.lif2(z2)
        pooled = spk2.mean(dim=-1)
        return self.head(pooled)

    def register_spike_hooks(self) -> list[SpikeActivationHook]:
        hook1 = SpikeActivationHook("lif1")
        hook2 = SpikeActivationHook("lif2")
        self.lif1.register_forward_hook(lambda m, i, o: hook1(m, i, o[0]))
        self.lif2.register_forward_hook(lambda m, i, o: hook2(m, i, o[0]))
        self._hooks = [hook1, hook2]
        return self._hooks

    def get_spike_rates(self) -> dict[str, float]:
        result = {}
        for hook in self._hooks:
            maps = hook.get_activation_map(clear=False)
            if maps is not None:
                result[hook.name] = float(maps.float().mean().item())
        return result


class CNN1D(nn.Module):
    """Non-spiking 1D CNN baseline for hybrid CNN/SNN comparison."""

    def __init__(self, input_shape: tuple[int, int], num_classes: int, hidden_channels: int = 32):
        super().__init__()
        _, channels = input_shape
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.net(x).squeeze(-1)
        return self.head(features)






def build_model(model_name, input_shape, num_classes, n_features=None):
    if model_name == 'cnn1d':
        return CNN1D(input_shape=input_shape, num_classes=num_classes)
    if model_name == 'spiking_cnn1d':
        return SpikingCNN1D(input_shape=input_shape, num_classes=num_classes)
    raise ValueError(f'Unknown model: {model_name}')
