import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def rmse(prediction, observation, original_std):
    prediction = prediction.squeeze()
    observation = observation.squeeze()

    assert original_std.dim() == 0
    assert prediction.shape == observation.shape
    assert len(prediction.shape) == 1

    result = torch.linalg.norm(observation - prediction)
    result = original_std * result / np.sqrt(len(observation))
    return result


def full_cov_nll(mean_vector, cov_matrix, observation):
    """
    Say we have (double check the shapes though, might as well `.squeeze()` all
    the tensors)
    > f_preds = model(xs)  # prediction without observation noise
    > y_preds = likelihood(model(xs))  # prediction with observation noise
    > mean_vector = y_preds.mean
    > cov_matrix = y_preds.covariance_matrix

    Then `full_cov_nll(mean_vector, cov_matrix, observation)` should be equal to
    > mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    > -mll(f_preds, observation)*len(observation)  # note: f_preds, not y_preds
    It should also be equal to
    > -y_preds.log_prob(observation)
    """
    mean_vector = mean_vector.squeeze()
    cov_matrix = cov_matrix.squeeze()
    observation = observation.squeeze()

    assert len(mean_vector.shape) == 1
    assert len(cov_matrix.shape) == 2
    assert len(observation.shape) == 1
    assert mean_vector.shape == observation.shape
    assert cov_matrix.shape[0] == cov_matrix.shape[1]
    assert cov_matrix.shape[0] == mean_vector.shape[0]

    distribution = MultivariateNormal(mean_vector, cov_matrix)
    return -distribution.log_prob(observation)


def diag_cov_nll(mean_vector, stds_vector, observation):
    """
    Say we have (double check the shapes though, might as well `.squeeze()` all
    the tensors)
    > f_preds = model(xs)  # prediction without observation noise
    > y_preds = likelihood(model(xs))  # prediction with observation noise
    > mean_vector = y_preds.mean
    > cov_matrix = y_preds.covariance_matrix

    Then `diag_cov_nll(mean_vector, stds_vector, observation)` should be equal
    to
    > torch.nn.functional.gaussian_nll_loss(
          mean_vector,
          observation,
          stds_vector**2, ## notice the power 2
          reduction="sum",
          full=True # this adds the "len(observation)/2*np.log(2*np.pi)" const
      ))

    """
    mean_vector = mean_vector.squeeze()
    stds_vector = stds_vector.squeeze()
    observation = observation.squeeze()

    assert len(mean_vector.shape) == 1
    assert len(stds_vector.shape) == 1
    assert len(observation.shape) == 1
    assert mean_vector.shape == stds_vector.shape
    assert stds_vector.shape == observation.shape

    distribution = MultivariateNormal(mean_vector, torch.diag(stds_vector**2))
    return -distribution.log_prob(observation)


def print_metrics(results_list, *, xs_train, ys_train, xs_test, ys_test, original_std):
    num_experiment_repetitions = len(results_list)

    rmse_list = np.zeros((num_experiment_repetitions,))
    rmse_train_list = np.zeros((num_experiment_repetitions,))
    diag_nll_list = np.zeros((num_experiment_repetitions,))
    diag_nll_list[:] = np.nan
    full_nll_list = np.zeros((num_experiment_repetitions,))
    full_nll_list[:] = np.nan

    no_uncertainty = False

    for i in range(num_experiment_repetitions):
        pred, std, test_cov = results_list[i]

        rmse_list[i] = rmse(
            pred[xs_test], observation=ys_test, original_std=original_std
        )
        rmse_train_list[i] = rmse(
            pred[xs_train], observation=ys_train, original_std=original_std
        )
        if (std is not None) and (test_cov is not None):
            diag_nll_list[i] = diag_cov_nll(
                pred[xs_test], std[xs_test], observation=ys_test
            )
            full_nll_list[i] = full_cov_nll(
                pred[xs_test], test_cov, observation=ys_test
            )
        else:
            no_uncertainty = True

    if diag_nll_list.min() == np.nan:
        best_ind = rmse_list.argmin()
    else:
        best_ind = diag_nll_list.argmin()

    best_pred, best_std, _ = results_list[best_ind]

    print(
        f"Test RMSE is {rmse_list.mean():.2f}"
        f" ± {rmse_list.std():.2f}"
        f" (best: {rmse_list.min():.2f})"
    )
    print(
        f"Train RMSE is {rmse_train_list.mean():.2f}" f" ± {rmse_train_list.std():.2f}"
    )

    if not no_uncertainty:
        print(
            f"Test diag-NLL is {diag_nll_list.mean():.2f}"
            f" ± {diag_nll_list.std():.2f}"
            f" (best: {diag_nll_list.min():.2f})"
        )
        print(
            f"Test full-NLL is {full_nll_list.mean():.2f}"
            f" ± {full_nll_list.std():.2f}"
            f" (best: {full_nll_list.min():.2f})"
        )

    return best_pred, best_std
