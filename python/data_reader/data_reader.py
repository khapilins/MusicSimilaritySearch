import tensorflow as tf


def create_data_reader(loader,
                       batch_size,
                       test_batch_size,
                       augmentations=None,
                       prefetch=128,
                       num_parallel_calls=4):
    with tf.name_scope('data_reader'):
        def gen(mode):
            while True:
                if mode.lower() == 'train':
                    yield loader.next_train()
                elif mode.lower() == 'test':
                    yield loader.next_test()
                else:
                    raise Exception('Unknown reader mode')

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: gen('train'),
            output_types=tuple(loader.get_types())).prefetch(prefetch)

        dataset_test = tf.data.Dataset.from_generator(
            generator=lambda: gen('test'),
            output_types=tuple(loader.get_types())).prefetch(prefetch)

        if augmentations is not None:
            for aug in augmentations:
                dataset = dataset.map(
                    aug,
                    num_parallel_calls=num_parallel_calls)

        dataset = dataset.padded_batch(batch_size, loader.get_shapes())
        dataset_test = dataset_test.padded_batch(test_batch_size,
                                                 loader.get_shapes())

        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)

        next_element = iterator.get_next()

        train_init_op = iterator.make_initializer(dataset)
        test_init_op = iterator.make_initializer(dataset_test)

        return (next_element, train_init_op, test_init_op)
