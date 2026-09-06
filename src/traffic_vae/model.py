"""A fully connected VAE with explicit loss scaling and deterministic scoring."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        layers = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(width, latent_dim)
        self.logvar = nn.Linear(width, latent_dim)
        layers = []
        width = latent_dim
        for hidden in reversed(hidden_dims):
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, sample: bool = True):
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-20.0, 20.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return self.decoder(z), mu, logvar


def loss_terms(reconstruction, x, mu, logvar, beta):
    reconstruction_loss = (reconstruction - x).square().sum(dim=1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    return reconstruction_loss + beta * kl, reconstruction_loss, kl


@torch.no_grad()
def score(model: VAE, values: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(values), batch_size):
        x = torch.from_numpy(values[start:start + batch_size])
        reconstruction, _, _ = model(x, sample=False)
        scores.append(torch.linalg.vector_norm(x - reconstruction, dim=1).numpy())
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite anomaly scores.")
    return result


def train(model: VAE, training: np.ndarray, validation: np.ndarray, config: dict, logger):
    loader = DataLoader(TensorDataset(torch.from_numpy(training)),
                        batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        totals = np.zeros(3)
        for (x,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(x)
            terms = loss_terms(reconstruction, x, mu, logvar, config["beta"])
            if not torch.isfinite(terms[0]):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            terms[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            totals += np.asarray([term.item() for term in terms]) * len(x)
        totals /= len(training)
        # Validation reconstruction at the posterior mean matches the detector score.
        validation_mse = float(np.mean(score(model, validation, config["batch_size"]).astype(float) ** 2))
        history.append({"epoch": epoch, "loss": totals[0], "reconstruction": totals[1],
                        "kl": totals[2], "validation_squared_l2": validation_mse})
        logger.info("Epoch %03d | loss %.6f | reconstruction %.6f | KL %.6f | validation %.6f",
                    epoch, *totals, validation_mse)
        if validation_mse < best:
            best, stale = validation_mse, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            logger.info("Early stopping at epoch %d", epoch)
            break
    model.load_state_dict(best_state)
    return history
