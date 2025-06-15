import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.util import predictive
from pyro.nn import PyroModule, PyroSample
from torch_geometric.nn import GCNConv


# a convinience utility
def glorot(fan_in, fan_out, scale):
    return scale * np.sqrt(2 / (fan_in + fan_out))


# Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html
class BayesianGCNModel(PyroModule):
    def __init__(
        self, hidden_layer_size=10, num_intermediate_layers=0, prior_scale=1.0
    ):
        super().__init__()

        self.first_layer = PyroModule[GCNConv](1, hidden_layer_size)
        self.first_layer.lin = PyroModule[nn.Linear](
            self.first_layer.in_channels, self.first_layer.out_channels, bias=False
        )
        self.first_layer.reset_parameters()
        first_layer_std = glorot(1, hidden_layer_size, prior_scale)
        self.first_layer.lin.weight = PyroSample(
            dist.Normal(0.0, first_layer_std).expand([hidden_layer_size, 1]).to_event(2)
        )
        self.first_layer.bias = PyroSample(
            dist.Normal(0.0, first_layer_std).expand([hidden_layer_size]).to_event(1)
        )

        self.last_layer = PyroModule[GCNConv](hidden_layer_size, 1)
        self.last_layer.lin = PyroModule[nn.Linear](
            self.last_layer.in_channels, self.last_layer.out_channels, bias=False
        )
        self.last_layer.reset_parameters()
        last_layer_std = glorot(hidden_layer_size, 1, prior_scale)
        self.last_layer.lin.weight = PyroSample(
            dist.Normal(0.0, last_layer_std).expand([1, hidden_layer_size]).to_event(2)
        )
        self.last_layer.bias = PyroSample(
            dist.Normal(0.0, last_layer_std).expand([1]).to_event(1)
        )

        # set up intermediate layers
        self.intermediate_layers = PyroModule[nn.ModuleList](
            [
                PyroModule[GCNConv](hidden_layer_size, hidden_layer_size)
                for i in range(num_intermediate_layers)
            ]
        )
        self.inner_lin_layers = PyroModule[nn.ModuleList](
            [
                PyroModule[nn.Linear](hidden_layer_size, hidden_layer_size, bias=False)
                for i in range(num_intermediate_layers)
            ]
        )
        for i in range(num_intermediate_layers):
            cur_layer = self.intermediate_layers[i]
            cur_layer.lin = self.inner_lin_layers[i]

            cur_layer.reset_parameters()

            intermediate_layer_std = glorot(
                hidden_layer_size, hidden_layer_size, prior_scale
            )
            cur_layer.lin.weight = PyroSample(
                dist.Normal(0.0, intermediate_layer_std)
                .expand([hidden_layer_size, hidden_layer_size])
                .to_event(2)
            )
            cur_layer.bias = PyroSample(
                dist.Normal(0.0, intermediate_layer_std)
                .expand([hidden_layer_size])
                .to_event(1)
            )

    def forward(self, x, edge_index, edge_weight, y=None, obs_mask=None):
        x = self.first_layer(x, edge_index, edge_weight)
        x = F.relu(x)
        for layer in self.intermediate_layers:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
        x = self.last_layer(x, edge_index, edge_weight)

        mu = x.squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1))  # Infer the response noise
        # sigma = pyro.sample("sigma", dist.Gamma(0.1, 0.1))  # Infer the response noise

        return pyro.sample(
            "obs", dist.Normal(mu, sigma * sigma), obs=y, obs_mask=obs_mask
        )
        # return pyro.sample("obs", dist.Normal(mu, sigma), obs=y, obs_mask=obs_mask)


def get_bnn_predictions_stds_and_test_cov(
    layer_width, num_intermediate_layers, pyg_data, xs_test, ys, train_mask
):

    model = BayesianGCNModel(
        layer_width, num_intermediate_layers=num_intermediate_layers
    )
    model.double()

    mcmc = MCMC(NUTS(model), num_samples=100, num_chains=1)

    # Run MCMC
    processed_ys = torch.nan_to_num(
        ys, -(10**6)
    ).squeeze()  # because pyro cannot handle NaNs
    processed_ys[xs_test] = -(10**6)  # hide test data
    mcmc.run(
        pyg_data.x, pyg_data.edge_index, pyg_data.edge_weight, processed_ys, train_mask
    )

    posterior_samples = predictive(
        model, mcmc.get_samples(), pyg_data.x, pyg_data.edge_index, pyg_data.edge_weight
    )["obs"].T

    predictions = torch.mean(posterior_samples, 1, keepdim=True)
    stds = torch.std(posterior_samples, 1, keepdim=True)

    test_cov = torch.cov(posterior_samples[xs_test.squeeze(), :])
    test_cov += torch.eye(test_cov.shape[0]) * 1e-4  # for regulization

    return predictions, stds, test_cov
