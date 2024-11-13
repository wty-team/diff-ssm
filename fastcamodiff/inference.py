import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from models.fast_camodiff import FastCamoDiff
from config.config import Config


def load_model(config, checkpoint_path):
    """Load trained model"""
    model = FastCamoDiff(config).to(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def prepare_image(image_path, config):
    """Prepare image for inference"""
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def inference(model, image, config):
    """Run inference"""
    with torch.no_grad():
        # Initial noise
        x = torch.randn_like(image)

        # Gradually denoise
        for t in range(config.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=config.device)

            # Predict noise
            pred_noise = model(x, t_batch)

            # Check early termination conditions
            if t > 0:
                # Calculate error metrics for object and background regions
                if calculate_early_termination(pred_noise, image):
                    break

            # Update x
            if t > 0:
                x = update_x(x, pred_noise, t, config)
            else:
                x = pred_noise

    return torch.sigmoid(x)


def calculate_early_termination(pred, original, threshold=0.1):
    """Check if denoising can be terminated early"""
    error = F.mse_loss(pred, original)
    return error < threshold


def update_x(x, pred_noise, t, config):
    """Update x during denoising process"""
    alpha_t = 1 - config.beta1_start
    alpha_prev = 1 - config.beta1_start if t > 1 else 1
    alpha_bar = alpha_t ** t
    alpha_bar_prev = alpha_prev ** (t - 1)

    sigma = (
                    (1 - alpha_bar_prev) / (1 - alpha_bar) * config.beta1_start
            ) ** 0.5

    mean = (
                   (alpha_bar_prev ** 0.5) * config.beta1_start * x +
                   ((1 - alpha_bar) ** 0.5) * (1 - alpha_bar_prev) * pred_noise
           ) / (1 - alpha_bar)

    noise = torch.randn_like(x)
    x = mean + sigma * noise

    return x


def main():
    # Initialize config
    config = Config()

    # Load model
    model = load_model(config, config.model_save_path + '/best_model.pth')

    # Prepare image
    image = prepare_image('path/to/test/image.jpg', config)
    image = image.to(config.device)

    # Run inference
    pred_mask = inference(model, image, config)

    # Save result
    pred_mask = pred_mask.cpu().squeeze().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(pred_mask).save('prediction.png')


if __name__ == '__main__':
    main()