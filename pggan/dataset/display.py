from dataset import TFRecordDataset


def display(tfrecord_dir_path: str):
    print(f'Loading dataset {tfrecord_dir_path}')

    dataset = TFRecordDataset(tfrecord_dir_path, max_label_size='full',
                              repeat=False, shuffle_mb=0)

    idx = 0
    while True:
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if idx == 0:
            print('Displaying images')
            import cv2  # pip install opencv-python
            cv2.namedWindow('dataset_tool')
            print('Press SPACE or ENTER to advance, ESC to exit')
        print('\nidx = %-8d\nlabel = %s' % (idx, labels[0].tolist()))
        cv2.imshow('dataset_tool', images[0].transpose(
            1, 2, 0)[:, :, ::-1])  # CHW => HWC, RGB => BGR
        idx += 1
        if cv2.waitKey() == 27:
            break
    print('\nDisplayed %d images.' % idx)
