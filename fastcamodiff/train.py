import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from config.config import Config
from models.fast_camodiff import FastCamoDiff
from data.dataset import get_dataloader
from utils.losses import TotalLoss
from utils.metrics import calculate_metrics


def train_one_epoch(model, train_loader, optimizer, loss_fn, config, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    with tqdm(total=num_batches) as progress_bar:
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Get data
            images = batch['image'].to(config.device)
            masks = batch['mask'].to(config.device)

            # Sample timestep and noise
            t = torch.randint(0, config.num_timesteps,
                              (images.shape[0],), device=config.device)
            noise = torch.randn_like(images)

            # Add noise to images
            noisy_images = add_noise(images, noise, t, config)

            # Predict noise
            pred_noise = model(noisy_images, t)

            # Calculate loss
            loss, loss_dict = loss_fn(pred_noise, noise, masks,
                                      config.beta1_start, config.beta2_start,
                                      noisy_images)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update progress
            total_loss += loss.item()
            progress_bar.set_description(
                f'Epoch {epoch} - Loss: {loss.item():.4f}'
            )
            progress_bar.update(1)

    return total_loss / num_batches


def validate(model, val_loader, loss_fn, config):
    model.eval()
    total_loss = 0
    metrics = {'s_measure': 0, 'f_measure': 0, 'mae': 0}
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(config.device)
            masks = batch['mask'].to(config.device)

            # Inference
            pred_masks = inference(model, images, config)

            # Calculate metrics
            batch_metrics = calculate_metrics(pred_masks, masks)
            for k, v in batch_metrics.items():
                metrics[k] += v

    # Average metrics
    for k in metrics:
        metrics[k] /= num_batches

    return metrics


def add_noise(x, noise, t, config):
    """Add noise to images according to diffusion schedule"""
    alpha_t = 1 - config.beta1_start
    alpha_bar = alpha_t ** t.view(-1, 1, 1, 1)
    return torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise


def main():
    # Initialize config
    config = Config()

    # Create model
    model = FastCamoDiff(config).to(config.device)

    # Create data loaders
    train_loader = get_dataloader(config, is_train=True)
    val_loader = get_dataloader(config, is_train=False)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Create loss function
    loss_fn = TotalLoss(config)

    # Training loop
    best_f_measure = 0

    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     loss_fn, config, epoch)

        # Validate
        metrics = validate(model, val_loader, loss_fn, config)

        # Log metrics
        logging.info(f'Epoch {epoch}:')
        logging.info(f'Train Loss: {train_loss:.4f}')
        logging.info(f'S-measure: {metrics["s_measure"]:.4f}')
        logging.info(f'F-measure: {metrics["f_measure"]:.4f}')
        logging.info(f'MAE: {metrics["mae"]:.4f}')

        # Save best model
        if metrics['f_measure'] > best_f_measure:
            best_f_measure = metrics['f_measure']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, f'{config.model_save_path}/best_model.pth')


if __name__ == '__main__':
    main()