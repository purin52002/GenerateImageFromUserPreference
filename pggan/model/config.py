from pathlib import Path

result_dir_path = Path('results')

G_arg_dict = dict(                                   # Generator architecture:

    # Overall multiplier for the number of feature maps.
    fmap_base=8192,
    # log2 of feature map reduction when doubling the resolution.
    fmap_decay=1.0,
    fmap_max=512,          # Maximum number of feature maps on any resolution.
    latent_size=512,          # Dimensionality of the latent vector.
    # Normalize latent vector to lie on the unit hypersphere?
    normalize_latents=True,
    use_wscale=True,         # Use equalized learning rate?
    use_pixelnorm=True,         # Use pixelwise normalization?
    use_leakyrelu=True,         # Use leaky ReLU?
    use_batchnorm=False,        # Use batch normalization?
    # Use tanh activation for the last layer? If so, how much to scale the output?
    tanh_at_end=None,
)

D_arg_dict = dict(                                   # Discriminator architecture:FF
    # Overall multiplier for the number of feature maps.
    fmap_base=8192,
    # log2 of feature map reduction when doubling the resolution.
    fmap_decay=1.0,
    fmap_max=512,          # Maximum number of feature maps on any resolution.
    # Which minibatch statistic to append as an additional feature map?
    mbstat_func='Tstdeps',
    mbstat_avg='all',        # Which dimensions to average the statistic over?
    # Use minibatch discrimination layer? If so, how many kernels should it have?
    mbdisc_kernels=None,
    use_wscale=True,         # Use equalized learning rate?
    use_gdrop=False,        # Include layers to inject multiplicative Gaussian noise?
    use_layernorm=False,        # Use layer normalization?
)
