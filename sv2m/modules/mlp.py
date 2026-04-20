import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        channel: int = 0,
        dropout: float = 0.5,
        use_bn: bool = True,
        momentum: float = 0.99,
        hidden_size: int = 1024,
        init_method: str = "xavier",
    ) -> None:
        super(MLP, self).__init__()
        self.init_method = init_method
        modules = []
        if hidden_size > 0:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=channel))
            modules.append(nn.ReLU())
            # modules.append(nn.Sigmoid())
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=channel, momentum=momentum))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=channel))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(output_size, output_size))
        self.net = nn.Sequential(*modules)
        # Initialize linear layer weights
        self.init_linear_weights(self.net, self.init_method)

    def init_linear_weights(self, model: nn.Module, init_method: str) -> None:
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return output

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
