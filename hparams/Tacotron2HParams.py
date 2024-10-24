from text import symbols, symbols_ro

class Tacotron2HParams:
    seed = 88

    # Distributed  Training
    num_gpus = 0
    dist_backend = "nccl"
    dist_url = "tcp://localhost:54321"
    world_size = 1

    # Data Parameters
    # text_cleaners = ["english_cleaners"] # English
    # text_cleaners = ["transliteration_cleaners"] # Romanian without diacritics
    text_cleaners = ["basic_cleaners"] # Romanian
    
    # Audio
    num_mels = 80
    num_freq = 513
    sample_rate = 22050
    frame_shift = 256
    frame_length = 1024
    fmin = 0
    fmax = 8000
    power = 1.5
    gl_iters = 30

    # Train
    is_cuda = True
    pin_mem = True
    n_workers = 4
    prep = True
    lr = 2e-3
    betas = (0.9, 0.999)
    eps = 1e-6
    sch_step = 4000
    max_iter = 500000
    batch_size = 32
    iters_per_log = 10
    iters_per_sample = 500
    iters_per_ckpt = 10000
    weight_decay = 1e-6
    grad_clip_thresh = 1.0

    # English and Romanian example text
    # eg_text = "OMAK is a thinking process which considers things always positively."
    eg_text = "Astfel, lucrarea mea de licență este finalizată."

    # Model Parameters
    n_symbols = len(symbols)
    n_symbols_ro = len(symbols_ro)
    symbols_embedding_dim = 512

    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512

    # Decoder parameters
    n_frames_per_step = 3
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_ratio = 10
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # Attention parameters
    attention_rnn_dim = 1024
    attention_dim = 128

    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5
