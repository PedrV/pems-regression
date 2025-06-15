import gpytorch
import lab as B  # for defining new kernels
import numpy as np
import torch
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import Graph

from pems_regression.utils import dcn


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, outputscale=1.0):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        kernel = gpytorch.kernels.ScaleKernel(kernel)
        kernel.outputscale = outputscale
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class InverseCosineKernel(MaternKarhunenLoeveKernel):
    """
    Can only be used with the _normalized_ graph Laplacian.

    Parameters, the "lengthscale" and "nu", are essentially ignored and are only
    used for inferring the types.
    """

    def __init__(self, space: Graph, normalize: bool = True):
        super().__init__(space, space.num_vertices, normalize)

    def _spectrum(
        self, s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric
    ) -> B.Numeric:
        return B.cos(B.cast(B.dtype(lengthscale), s) * np.pi / 4)


class RandomWalkKernel(MaternKarhunenLoeveKernel):
    """
    Can only be used with the _normalized_ graph Laplacian.

    Parameters, the "lengthscale" and "nu", are essentially ignored and are only
    used for inferring the types.
    """

    def __init__(self, space: Graph, normalize: bool = True):
        super().__init__(space, space.num_vertices, normalize)

    def _spectrum(
        self, s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric
    ) -> B.Numeric:
        return (2 + lengthscale - B.cast(B.dtype(lengthscale), s)) ** nu


def get_gp_predictions_stds_and_test_cov(
    kernel, train_iterations=200, *, xs, ys, xs_train, ys_train, xs_test
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(0.01)  # initialization of observation noise
    # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.LessThan(0.2**2))
    # likelihood.noise = torch.tensor(0.1**2) # initialization of observation noise

    model = ExactGPModel(xs_train.squeeze(), ys_train.squeeze(), likelihood, kernel)
    model.double()
    likelihood.double()

    print("Initial gp_model:")
    print("kernel.base_kernel.nu =", dcn(model.covar_module.base_kernel.nu))
    print(
        "kernel.base_kernel.lengthscale =",
        dcn(model.covar_module.base_kernel.lengthscale),
    )
    print("kernel.outputscale =", dcn(model.covar_module.outputscale))
    print("likelihood.obs_noise =", dcn(model.likelihood.noise))
    print("")

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for i in range(train_iterations):
        optimizer.zero_grad()
        output = model(xs_train.squeeze())
        loss = -mll(output, ys_train.squeeze())
        loss.backward()
        if i == 0 or (i + 1) % 20 == 0:
            print("Iter %d/%d - Loss: %.5f" % (i + 1, train_iterations, loss.item()))
        optimizer.step()

    print("")
    print("Final model:")
    print(
        "kernel.base_kernel.nu =",
        model.covar_module.base_kernel.nu.detach().cpu().numpy(),
    )
    print(
        "kernel.base_kernel.lengthscale =",
        model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
    )
    print("kernel.outputscale =", model.covar_module.outputscale.detach().cpu().numpy())
    print("likelihood.obs_noise =", model.likelihood.noise.detach().cpu().numpy())

    model.eval()
    likelihood.eval()

    predictions = likelihood(model(xs.squeeze())).mean.squeeze()
    stds = likelihood(model(xs.squeeze())).stddev.squeeze()
    test_cov = likelihood(model(xs_test.squeeze())).covariance_matrix.squeeze()

    return predictions, stds, test_cov
