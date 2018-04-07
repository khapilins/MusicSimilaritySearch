import tensorflow as tf


def create_data_reader(loader,
                       batch_size,
                       augmentations=None,
                       prefetch=128,
                       num_parallel_calls=4):

    def gen():
        while True:
            yield loader.next()

    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=tuple(loader.get_types())).prefetch(prefetch)

    if augmentations is not None:
        for aug in augmentations:
            dataset = dataset.map(
                aug,
                num_parallel_calls=num_parallel_calls)

    return dataset.padded_batch(batch_size, loader.get_shapes())\
        .make_one_shot_iterator().get_next()
