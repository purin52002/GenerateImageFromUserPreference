data_dir_path = 'datasets'
result_dir_path = 'results'

random_seed = 1000
dataset = None

# Training parameters:
train = dict(
    # Main training func.
    func='train_gan',
    # Alternate between training generator and discriminator?
    separate_funcs=True,
    # n_{critic}
    D_training_repeats=1,
    # \alpha
    G_learning_rate_max=0.001,
    # \alpha
    D_learning_rate_max=0.001,
    # Exponential running average of generator weights.
    G_smoothing=0.999,
    # \beta_1
    adam_beta1=0.0,
    # \beta_2
    adam_beta2=0.99,
    # \epsilon
    adam_epsilon=1e-8,
    # Minibatch size for low resolutions.
    minibatch_default=16,
    # Minibatch sizes for high resolutions.
    minibatch_overrides={256: 14, 512: 6,  1024: 3},
    # Learning rate rampup.
    rampup_kimg=40,
    # Learning rate rampdown.
    rampdown_kimg=0,
    # Network resolution at the beginning.
    lod_initial_resolution=4,
    # Thousands of real images to show before doubling network resolution.
    lod_training_kimg=600,
    # Thousands of real images to show when fading in new layers.
    lod_transition_kimg=600,
    # Thousands of real images to show in total.
    total_kimg=15000,
    # Do not inject multiplicative Gaussian noise in the discriminator.
    gdrop_coef=0.0,
)

# Generator architecture:
G = dict(
    # Configurable network template.
    func='G_paper',
    # Overall multiplier for the number of feature maps.
    fmap_base=8192,
    # log2 of feature map reduction when doubling the resolution.
    fmap_decay=1.0,
    # Maximum number of feature maps on any resolution.
    fmap_max=512,
    # Dimensionality of the latent vector.
    latent_size=512,
    # Normalize latent vector to lie on the unit hypersphere?
    normalize_latents=True,
    # Use equalized learning rate?
    use_wscale=True,
    # Use pixelwise normalization?
    use_pixelnorm=True,
    # Use leaky ReLU?
    use_leakyrelu=True,
    # Use batch normalization?
    use_batchnorm=False,
    # Use tanh activation for the last layer?
    #  If so, how much to scale the output?
    tanh_at_end=None,
)
# Discriminator architecture:
D = dict(
    # Configurable network template.
    func='D_paper',
    # Overall multiplier for the number of feature maps.
    fmap_base=8192,
    # log2 of feature map reduction when doubling the resolution.
    fmap_decay=1.0,
    # Maximum number of feature maps on any resolution.
    fmap_max=512,
    # Which minibatch statistic to append as an additional feature map?
    mbstat_func='Tstdeps',
    # Which dimensions to average the statistic over?
    mbstat_avg='all',
    # Use minibatch discrimination layer?
    # If so, how many kernels should it have?
    mbdisc_kernels=None,
    # Use equalized learning rate?
    use_wscale=True,
    # Include layers to inject multiplicative Gaussian noise?
    use_gdrop=False,
    # Use layer normalization?
    use_layernorm=False,
)
# Loss function:
loss = dict(
    # Wasserstein (WGAN).
    type='iwass',
    # \lambda
    iwass_lambda=10.0,
    # \epsilon_{drift}
    iwass_epsilon=0.001,
    # \alpha
    iwass_target=1.0,
    # AC-GAN
    cond_type='acgan',
    # Weight of the conditioning terms.
    cond_weight=1.0,
)

if 1:
    run_desc = 'celeba'

    dataset = dict(h5_path='celeba-128x128.h5', resolution=128,
                   max_labels=0, mirror_augment=True)

    train.update(lod_training_kimg=800, lod_transition_kimg=800,
                 rampup_kimg=0, total_kimg=10000, minibatch_overrides={})
    G.update(fmap_base=2048)
    D.update(fmap_base=2048)
