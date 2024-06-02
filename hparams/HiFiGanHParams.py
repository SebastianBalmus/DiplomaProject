class HiFiGanHParams:
    seed = 88

    # Distributed  Training
    num_gpus = 0
    dist_backend = "nccl"
    dist_url = "tcp://localhost:54321"
    world_size = 1
    num_workers = 4

    # Audio
    segment_size = 8192
    num_mels = 80
    num_freq = 1025
    n_fft = 1024
    hop_size = 256
    win_size = 1024
    sampling_rate = 22050
    fmin = 0
    fmax = 8000
    fmax_for_loss = None
    max_wav_value = 32768.0

    # Train
    batch_size = 16
    learning_rate = 0.0002
    betas = (0.9, 0.999)  # adam_b1, adam_b2
    lr_decay = 0.999
    training_epochs = 3100
    stdout_interval = 5
    checkpoint_interval = 5000
    summary_interval = 100
    validation_interval = 1000

    # Model Parameters
    upsample_rates = [8, 8, 2, 2]
    upsample_kernel_sizes = [16, 16, 4, 4]
    upsample_initial_channel = 128
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    lrelu_slope = 0.1
