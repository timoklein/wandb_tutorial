# Code is taken and modified from https://github.com/pytorch/examples/tree/main/vae
from pathlib import Path
from datetime import datetime

import tqdm
import wandb
import pyrallis
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from set_device import set_cuda_configuration
from model import VAE
from config import Config


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


def train(epoch: int, model: VAE, train_loader: torch.utils.data.DataLoader, optimizer: optim.Adam, device: torch.device):
    model.train()
    train_loss = 0
    bar = tqdm.tqdm(train_loader, leave=False)
    for batch_idx, (data, _) in enumerate(bar):
        train_step = (epoch - 1) * len(train_loader) + batch_idx
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        reconstruction_loss, kl_loss = loss_function(recon_batch, data, mu, logvar)
        total_loss = reconstruction_loss + kl_loss
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()
        # TODO: Log train loss to wandb
        wandb.log(
            {
                "train/step": train_step,
                "train/total_loss": total_loss.item(),
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/kl_loss": kl_loss.item(),
                "train/mu": wandb.Histogram(mu.detach().view(-1), num_bins=10),
                "train/std": wandb.Histogram(logvar.detach().exp().view(-1), num_bins=10),
            },
        )
        bar.set_description(f"Train Epoch: {epoch} \tLoss: {total_loss.item() / len(data):.6f}")

    return train_loss / len(train_loader.dataset)


@torch.inference_mode()
def test(
    epoch: int,
    model: VAE,
    test_loader: torch.utils.data.DataLoader,
    bs: int,
    device: torch.device,
    img_path: Path,
    test_interval: int,
):
    model.eval()
    test_loss = 0
    bar = tqdm.tqdm(test_loader, leave=False)
    for i, (data, _) in enumerate(bar):
        test_step = ((epoch - 1) // test_interval) * len(test_loader) + i
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        # TODO: Log the test loss to wandb
        reconstruction_loss, kl_loss = loss_function(recon_batch, data, mu, logvar)
        total_loss = reconstruction_loss + kl_loss
        test_loss += total_loss.item()

        test_logs = {
            "test/step": test_step,
            "test/total_loss": total_loss.item(),
            "test/reconstruction_loss": reconstruction_loss.item(),
            "test/kl_loss": kl_loss.item(),
        }

        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(bs, 1, 28, 28)[:n]])
            # TODO: Log the image to wandb
            save_image(comparison.cpu(), img_path / f"reconstruction_{str(epoch)}.png", nrow=n)
            test_logs["test/comparison"] = wandb.Image(comparison.cpu())
        wandb.log(test_logs)
        bar.set_description(f"Test Epoch: {epoch} \tLoss: {total_loss.item() / len(data):.6f}")

    return test_loss / len(test_loader.dataset)


def main(cfg: Config):
    device = set_cuda_configuration(cfg.gpu)

    # NOTE: Deactivate to generate different results
    # torch.use_deterministic_algorithms(True)
    # torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)

    kwargs = {"num_workers": 1, "pin_memory": True} if device.type != "cpu" else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=cfg.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=False, transform=transforms.ToTensor()),
        batch_size=cfg.batch_size,
        shuffle=False,
        **kwargs,
    )

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    img_path = Path("imgs")
    img_path.mkdir(exist_ok=True, parents=True)
    model_path = Path("models")
    model_path.mkdir(exist_ok=True, parents=True)

    # TODO: Initialize wandb
    # TODO: Set up for hyperparameter optimization
    timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg is None:
        # Running hyperparameter optimization, so the config is just a dummy
        run = wandb.init()
        cfg = Config(**wandb.config)
    else:
        run = wandb.init(
            entity=None,
            project="Wandb Tutorial",
            name=f"{timestamp}__VAE",
            config=vars(cfg),
            mode="online" if cfg.wandb else "disabled",
        )

    # TODO: Set up for logging with different x axes
    wandb.define_metric("test/step")
    wandb.define_metric("test/*", step_metric="test/step")
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    bar = tqdm.trange(1, cfg.epochs + 1, desc="Epochs", leave=True)
    for epoch in bar:
        train_loss = train(epoch, model, train_loader, optimizer, device)
        bar.set_description(f"Epoch: {epoch} | Train Loss: {train_loss / len(train_loader.dataset):.4f}")
        if epoch % cfg.eval_interval == 0:
            test_loss = test(epoch, model, test_loader, cfg.batch_size, device, img_path, test_interval=cfg.eval_interval)
            bar.set_description(f"Epoch: {epoch} | Test Loss: {test_loss / len(test_loader.dataset):.4f}")
        # Use only when you're doing commit=False in the other wandb.log calls
        # wandb.log({"epoch": epoch}, commit=True)
        with torch.inference_mode():
            # Generate a sample and save it
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), img_path / f"sample_{str(epoch)}.png")

    # TODO: Call wandb.finish() here
    # TODO: Save a model as example
    torch.save(model.state_dict(), model_path / f"VAE_{timestamp}.pt")
    wandb.save(str(model_path / f"VAE_{timestamp}.pt"), policy="now")


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=Config)
    main(cfg)
