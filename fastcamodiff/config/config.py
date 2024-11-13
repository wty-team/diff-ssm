class Config:
    # Dataset settings
    img_size = 256
    batch_size = 6

    # Model parameters
    hidden_dim = 192
    expanded_dim = 384  # 2 * hidden_dim
    num_blocks = 12
    ssm_dim = 16

    # Diffusion settings
    num_timesteps = 1000
    beta1_start = 0.5  # Initial beta1 for object region
    beta2_start = 0.5  # Initial beta2 for background region

    # Training settings
    learning_rate = 1e-5
    num_epochs = 100

    # Early termination thresholds
    object_error_threshold = 0.1
    background_error_threshold = 0.05

    # Hardware settings
    device = "cuda"  # or "cpu"
    num_workers = 4

    # Paths
    train_data = {
        'camo': './datasets/CAMO/Train/',
        'cod10k': './datasets/COD10K/Train/',
        'nc4k': './datasets/NC4K/Train/'
    }

    val_data = {
        'camo': './datasets/CAMO/Test/',
        'cod10k': './datasets/COD10K/Test/',
        'nc4k': './datasets/NC4K/Test/'
    }

    # 数据集组织方式
    data_structure = {
        'camo': {'image': 'imgs', 'mask': 'masks'},
        'cod10k': {'image': 'Image', 'mask': 'GT'},
        'nc4k': {'image': 'imgs', 'mask': 'masks'}
    }

    # Metrics
    alpha = 0.5  # For S-measure
    beta = 0.3  # For F-measure
