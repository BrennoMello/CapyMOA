from capymoa.evaluation.ocl import (
    ocl_train_eval_loop, ocl_delay_train_eval_loop)
from capymoa.type_alias import LabelProbabilities
from torch.nn.functional import cross_entropy
from capymoa.datasets.ocl import SplitMNIST
from capymoa.base import BatchClassifier
from capymoa.instance import Instance
from capymoa.stream import Schema
from typing import Tuple
from torch import Tensor
from torch import nn
import numpy as np
import torch


class ReservoirSampler:
    def __init__(self, item_count: int, feature_count: int):
        self.item_count = item_count
        self.feature_count = feature_count
        self.reservoir_x = torch.zeros((item_count, feature_count))
        self.reservoir_y = torch.zeros((item_count,), dtype=torch.long)
        self.count = 0

    def update(self, x: Tensor, y: Tensor) -> None:
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self.feature_count,
        )
        assert y.shape == (batch_size,)

        for i in range(batch_size):
            if self.count < self.item_count:
                # Fill the reservoir
                self.reservoir_x[self.count] = x[i]
                self.reservoir_y[self.count] = y[i]
            else:
                # Reservoir sampling
                index = torch.randint(0, self.count + 1, (1,))
                if index < self.item_count:
                    self.reservoir_x[index] = x[i]
                    self.reservoir_y[index] = y[i]
            self.count += 1

    def sample_n(self, n: int) -> Tuple[Tensor, Tensor]:
        indices = torch.randint(0, min(self.count, self.item_count), (n,))
        return self.reservoir_x[indices], self.reservoir_y[indices]


class ExperienceReplay(BatchClassifier):
    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        reservoir_size: int,
        batch_size: int,
        learning_rate: float,
        device: str = "cpu",
    ):
        super().__init__(schema=schema, batch_size=batch_size)
        self.reservoir = ReservoirSampler(reservoir_size, schema.get_num_attributes())
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

    def batch_train(self, x: np.ndarray, y: np.ndarray):
        x: Tensor = torch.from_numpy(x)
        y: Tensor = torch.from_numpy(y).long()

        self.reservoir.update(x, y)

        replay_x, replay_y = self.reservoir.sample_n(self.batch_size)
        train_x = torch.cat((x, replay_x), dim=0).to(self.device)
        train_y = torch.cat((y, replay_y), dim=0).to(self.device)

        self.optimizer.zero_grad()
        y_hat = self.model(train_x)
        loss = cross_entropy(y_hat, train_y)
        loss.backward()
        self.optimizer.step()

    @torch.no_grad
    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        x = torch.from_numpy(instance.x).to(self.device)
        y_hat: Tensor = self.model.forward(x)
        return y_hat.softmax(dim=0).cpu().numpy()

    def __str__(self) -> str:
        return "ExperienceReplay"

class SimpleMLP(nn.Module):
    def __init__(self, schema: Schema, hidden_size: int):
        super().__init__()
        num_classes = schema.get_num_classes()

        self.fc1 = nn.Linear(schema.get_num_attributes(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)
        self.fc2 = nn.utils.parametrizations.weight_norm(self.fc2, name="weight")
        weight_g = self.fc2.parametrizations.weight.original0
        # Set the magnitude to the unit vector
        weight_g.requires_grad_(False).fill_(1.0 / (num_classes**0.5))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run():
    
    stream = SplitMNIST(num_tasks=10 , shuffle_tasks=True)
    mlp = SimpleMLP(stream.schema, 64)
    learner = ExperienceReplay(
        stream.schema,
        mlp,
        reservoir_size=200,
        batch_size=64,
        learning_rate=0.01,
        device="cpu",
    )
    # r = ocl_train_eval_loop(
    #     learner,
    #     stream.train_streams,
    #     stream.test_streams,
    #     continual_evaluations=10,
    #     progress_bar=True,
    # )
    r = ocl_delay_train_eval_loop(
        learner,
        stream.train_streams,
        stream.test_streams,
        continual_evaluations=10,
        delay=500,
        progress_bar=True,
    )
    print(f"Forward Transfer  {r.forward_transfer:.2f}")
    print(f"Backward Transfer {r.backward_transfer:.2f}")
    print(f"Accuracy          {r.accuracy_final:.2f}")
    print(f"Online Accuracy   {r.prequential_cumulative_accuracy:.2f}")


if __name__ == "__main__":
    run()