def free_energy(x1, y_obs, beta, sigma_obs=0.1):
    kl_term = 0.5 * (beta ** 2).sum()
    recon_term = 0.5 / (sigma_obs ** 2) * ((y_obs - x1) ** 2).mean()
    return kl_term + recon_term