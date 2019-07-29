dataset_arg_dict = {
    'tfrecord_dir_path': '',
    # Dataset resolution, None = autodetect.
    'resolution': None,
    # Relative path of the labels file, None = autodetect.
    'label_file': None,
    # 0 = no labels, 'full' = full labels,
    # <int> = N first label components.
    'max_label_size': 0,
    # Repeat dataset indefinitely.
    'repeat': True,
    # Shuffle data within specified window (megabytes),
    # 0 = disable shuffling.
    'shuffle_mb': 4096,
    # Amount of data to prefetch (megabytes),
    # 0 = disable prefetching.
    'prefetch_mb': 2048,
    # Read buffer size (megabytes).
    'buffer_mb': 256,
    # Number of concurrent threads.
    'num_threads': 2
}
