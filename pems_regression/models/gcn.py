import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNModel(torch.nn.Module):
    def __init__(
        self, hidden_layer_size=10, num_intermediate_layers=0, input_dropout=False
    ):
        super().__init__()
        self.input_dropout = input_dropout

        self.first_layer = GCNConv(1, hidden_layer_size)

        self.intermediate_layers = [
            GCNConv(hidden_layer_size, hidden_layer_size)
            for _ in range(num_intermediate_layers)
        ]

        self.last_layer = GCNConv(hidden_layer_size, 1)

    def forward(self, x, edge_index, edge_weight):
        if self.input_dropout:
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.first_layer(x, edge_index, edge_weight)
        x = F.relu(x)

        for layer in self.intermediate_layers:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)

        x = self.last_layer(x, edge_index, edge_weight)

        return x


def train(
    data, model, optimizer, criterion, epochs=500, *, only_print_end_result=False
):
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear gradients.
        outputs = model(
            data.x, data.edge_index, data.edge_weight
        ).squeeze()  # Perform a single forward pass.
        loss = criterion(
            outputs[data.train_mask], data.y.squeeze()[data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        if (not only_print_end_result and (epoch % 100) == 0) or (
            only_print_end_result and epoch + 1 == epochs
        ):
            print(f"Epochs:{epoch + 1:5d} | " f"Cur loss: {loss:.10f}")


def get_gcnn_predictions(
    layer_width,
    num_intermediate_layers,
    *,
    pyg_data,
    only_print_end_result=False,
    num_training_iterations=500,
):
    model = GCNModel(layer_width, num_intermediate_layers=num_intermediate_layers)
    model.double()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.MSELoss()
    model.train()
    train(
        pyg_data,
        model,
        optimizer,
        criterion,
        num_training_iterations,
        only_print_end_result=only_print_end_result,
    )
    model.eval()

    predictions = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_weight)

    return predictions


def get_ensemble_predictions_stds_and_test_cov(
    num_models, *, xs_test, ys, pyg_data, layer_width=100, num_intermediate_layers=7
):
    individual_model_predictions = torch.empty((len(ys), num_models))
    for i in range(num_models):
        individual_model_predictions[:, i] = get_gcnn_predictions(
            layer_width,
            num_intermediate_layers,
            pyg_data=pyg_data,
            only_print_end_result=True,
        ).squeeze()

    predictions = torch.mean(individual_model_predictions, 1, keepdim=True)
    stds = torch.std(individual_model_predictions, 1, keepdim=True)

    test_cov = torch.cov(individual_model_predictions[xs_test.squeeze(), :])
    test_cov += torch.eye(test_cov.shape[0]) * 1e-4  # for regulization

    return predictions, stds, test_cov
