# models/euler.py
import torch
import torch.nn as nn


class EulerSDE(nn.Module):

    def __init__(self, d, model_drift, variational_drift, diffusion):
        super().__init__()
        self.d = d
        self.model_drift = model_drift
        self.variational_drift = variational_drift
        self.diffusion = diffusion

    def forward(self, x0, n_steps, **context):

        dt = 1.0 / n_steps
        x = x0.clone()
        kl = torch.zeros(x0.shape[0], device=x0.device)

        for i in range(n_steps):
            t = torch.tensor(i * dt)
            b = self.model_drift(t, x)
            b_tilde = self.variational_drift(t, x, **context)
            sigma = self.diffusion(t, x)
            kl = kl + 0.5 * dt * (b_tilde ** 2).sum(-1)
            x = x + dt * (b + b_tilde) + sigma * dt**0.5 * torch.randn_like(x)

        return x, kl.mean()


def free_energy(x1, kl, y_obs, sigma_obs=0.1):
    recon_term = 0.5 / (sigma_obs ** 2) * ((y_obs - x1) ** 2).sum(-1).mean()
    return kl + recon_term