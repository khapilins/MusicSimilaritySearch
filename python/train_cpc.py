import os
import numpy as np
import tensorflow as tf
# I guess it's just easier for projections
# than plain tensorboard
from tensorboardX import SummaryWriter
import argparse
from models import get_model
from data_reader import create_data_reader, get_loader, get_augmentation
from utils import load, save
import json
import time
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True,
                        help='Where to store logs and checkpoints')
    parser.add_argument('--loader_params', type=str, required=True,
                        help='JSON file location with specified loader'
                        ' and its parameters')
    parser.add_argument('--model_params', type=str,
                        help='JSON file location with mpdel parameters')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load, if not specified,'
                        ' the latest checkpoint in logdir is used')
    parser.add_argument('--test_every', type=int, default=100,
                        help='Compute test loss and accuracy '
                        'every --test_every steps')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save model weights every --save_every')
    parser.add_argument('--summaries_every', type=int, default=1500,
                        help='Writes summaries on disk every summaries_every')
    parser.add_argument('--embeddings_every', type=int, default=1500,
                        help='Writes embeddings projections on disk every '
                             'embeddings_every')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=int(1e5) + 1)
    parser.add_argument('--n_losses', type=int, default=20,
                        help='Number of future-predicting losses')
    parser.add_argument('--l2_coeff', type=float, default=0,
                        help='L2 regularization for zs and cs embeddings '
                             'regularization')
    parser.add_argument('--n_z_units', type=float, default=128,
                        help='Number of units to project zs to before '
                        'dot-product with cs')
    parser.add_argument('--profile', action='store_true',
                        help='If selected, will output informaiton about '
                        'running time and memory into tensorboard graph '
                        'view alongside with other summaries. (so it\'s also '
                        'depends on --summaries_every. Note that it actually '
                        'quite expensive and somewhat decreases overall '
                        'perfomance even when not writing profiling info '
                        'during current step. BTW it\'s also takes a lot of '
                        'time to open in tensorboard as well')

    return parser.parse_args()


def get_f_predictions(zs, cs, k, n_units=128):
    # delete first k zs, to predict zs_t+k using c_t
    # delete last k cs, to match shapes
    with tf.variable_scope('f_prediction_' + str(k)):
        if k == 0:
            shifted_cs = cs
            shifted_zs = zs
        else:
            shifted_zs = zs[:, k:, :]
            shifted_cs = cs[:, :-k, :]
        zs_cs_concat = tf.concat((shifted_zs, shifted_cs), 2)

        zs_shape = zs.get_shape().as_list()
        cs_shape = cs.get_shape().as_list()

        prediction_layer1 = tf.layers.Dense(
            n_units,
            activation=tf.nn.tanh,
            name='f_prediction_layer1_' + str(k))
        prediction_layer2 = tf.layers.Dense(
            1,
            name='f_prediction_layer2_' + str(k))
        z_transformed = prediction_layer2(prediction_layer1(shifted_zs))
        f_pred = tf.reduce_sum(
            tf.multiply(z_transformed, shifted_cs), 2, keep_dims=True)

    with tf.variable_scope('contrastive_example_' + str(k)):
        z_contr_transformed = tf.concat((z_transformed[1:, :, :],
                                         z_transformed[:1, :, :]),
                                        0)
        contr_f_pred = tf.reduce_sum(
            tf.multiply(z_contr_transformed, shifted_cs), 2, keep_dims=True)
        contrastive_f_pred = tf.reshape(contr_f_pred, [-1])
    with tf.variable_scope('accuracy_' + str(k)):
        broadcasted_shape = (
            tf.shape(f_pred)[0],
            tf.shape(f_pred)[1],
            tf.shape(contrastive_f_pred)[-1])
        acc = (tf.reduce_sum(
                tf.where(
                    tf.math.greater(f_pred, contrastive_f_pred),
                    tf.ones(broadcasted_shape, dtype=tf.float32),
                    tf.zeros(broadcasted_shape, dtype=tf.float32))) /
               tf.reduce_prod(tf.cast(broadcasted_shape, tf.float32)))

    return f_pred, contrastive_f_pred, acc


def crop_inputs(im, size):
    return tf.cond(
        tf.greater(tf.shape(im)[0], size),
        lambda: tf.random_crop(im, [size, tf.shape(im)[-1]]),
        lambda: tf.convert_to_tensor(im, tf.float32))


if __name__ == '__main__':
    args = get_args()
    with open(args.loader_params) as f:
        loader_params = json.load(f)
    with open(args.model_params) as f:
        model_params = json.load(f)
    loader = get_loader(loader_params['loader_type'])(
        **loader_params['loader_params'])
    model = get_model(model_params['model_type'])(
        **model_params['model_params'])

    train_augmentations, test_augmentations = [], []
    if loader_params.get('augmentations', None):
        for aug in loader_params['augmentations']['train']:
            train_augmentations.append(
                get_augmentation(aug[0])(**aug[1]))
        for aug in loader_params['augmentations']['test']:
            test_augmentations.append(
                get_augmentation(aug[0])(**aug[1]))
        augmentations = {'train': train_augmentations,
                         'test': test_augmentations}
    else:
        augmentations = None

    reader, train_data_init_op, test_data_init_op = create_data_reader(
        loader, args.batch_size, args.test_batch_size,
        augmentations=augmentations)

    next_element = reader['clean_batch']
    next_element_aug = reader.get('aug_batch', None)
    metadata_labels = reader.get('metadata', reader['path'])

    next_element = tf.expand_dims(next_element, axis=2)

    model_out = model.create_model(next_element)

    writer = tf.summary.FileWriter(args.logdir,
                                   tf.get_default_graph())

    writerX = SummaryWriter(os.path.join(args.logdir, 'projections'))

    losses = []
    losses_summary = []
    test_losses_summary = []
    MIs = []

    for i in range(args.n_losses):
        with tf.name_scope('loss_' + str(i)):
            f_pred, contrastive_f_pred, acc = get_f_predictions(
                model.zs, model.cts, i)

            pred_shape = tf.shape(f_pred)
            batch_size = tf.cast(pred_shape[0], tf.float32)
            batch = batch_size
            time_steps = tf.cast(pred_shape[1], tf.float32)

            contrastive_f_pred = tf.reshape(
                tf.tile(contrastive_f_pred, [pred_shape[0]]),
                (batch_size, time_steps, -1))

            labels = tf.reshape(
                tf.tile(
                    tf.concat(([1], tf.zeros((batch_size, ))), 0),
                    [pred_shape[0] * pred_shape[1]]),
                (batch_size, pred_shape[1], -1))
            logits = tf.concat((f_pred, contrastive_f_pred), 2)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=labels, logits=logits, dim=2))
            MI = (tf.log(time_steps) - loss) / batch

            losses_summary.append(tf.summary.scalar('loss_k_' + str(i), loss))
            losses_summary.append(tf.summary.scalar('MI_k_' + str(i), MI))
            losses_summary.append(
                tf.summary.scalar('f_pred_p_' + str(i),
                                  tf.reduce_mean(f_pred)))

            losses_summary.append(
                tf.summary.scalar('contr_f_pred_' + str(i),
                                  tf.reduce_mean(contrastive_f_pred)))

            losses_summary.append(
                tf.summary.scalar('sum_contr_f_pred_' + str(i),
                                  tf.reduce_sum(contrastive_f_pred)))

            losses_summary.append(
                tf.summary.scalar('acc_' + str(i),
                                  acc))

            losses_summary.append(
                tf.summary.scalar(
                    'ratio_contr_f_pred_' + str(i),
                    tf.reduce_mean(
                        f_pred/tf.reduce_sum(contrastive_f_pred))))

            test_losses_summary.append(
                tf.summary.scalar('test_loss_k_' + str(i), loss))

            test_losses_summary.append(
                tf.summary.scalar('test_MI_k_' + str(i), MI))

            losses.append(loss)
            MIs.append(MI)

    with tf.name_scope('l2_norms'):
        zs_norm = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(model.zs), axis=2))
        cts_norm = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(model.cts), axis=2))
        losses_summary.append(tf.summary.scalar('l2_zs', zs_norm))
        losses_summary.append(tf.summary.scalar('l2_cs', cts_norm))

    with tf.name_scope('total_loss'):
        total_loss = sum(losses)
        if args.l2_coeff > 0:
            total_loss += args.l2_coeff * zs_norm
            total_loss += args.l2_coeff * cts_norm

        losses_summary.append(tf.summary.scalar('total_loss', total_loss))
        test_losses_summary.append(
            tf.summary.scalar('total_test_loss', total_loss))
        mean_MI = tf.reduce_mean(MIs)

    sums_op = tf.summary.merge(losses_summary)
    test_sums_op = tf.summary.merge(test_losses_summary)
    optimizer = tf.train.AdamOptimizer(args.lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = optimizer.minimize(total_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(train_data_init_op)
    saver = tf.train.Saver()
    ckpt_step = load(saver, sess, args.logdir, args.ckpt)

    pbar = tqdm(range(ckpt_step, args.n_steps))
    for i in pbar:
        try:
            if args.profile and i % args.summaries_every == 0:
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE,
                    report_tensor_allocations_upon_oom=True)
            else:
                run_metadata = None
                options = None
            loss_, mean_MI_, sums, zs, cs, ls, _ = sess.run([
                total_loss,
                mean_MI,
                sums_op,
                model.zs,
                model.cts,
                metadata_labels,
                opt],
                run_metadata=run_metadata,
                options=options)
            pbar.set_description(
                'Loss: {:.3f}, MI: {:.3f}'
                .format(loss_, mean_MI_))

            if i % args.summaries_every == 0:
                writer.add_summary(sums, i)
                if args.profile:
                    writer.add_run_metadata(
                        run_metadata, 'run_meta_step_' + str(i), i)

            if i % args.test_every == 0:
                sess.run(test_data_init_op)
                loss_, mean_MI_, sums, tzs, tcs, tls = sess.run([
                    total_loss,
                    mean_MI,
                    test_sums_op,
                    model.zs,
                    model.cts,
                    metadata_labels])
                print('Test loss: {:3f}, test MI: {:3f}'.
                      format(loss_, mean_MI_))
                writer.add_summary(sums, i)
                sess.run(train_data_init_op)
            if i % args.save_every == 0:
                save(saver, sess, args.logdir, i)

            if i % args.embeddings_every == 0:
                print('Saving projections')
                tzs = np.squeeze(tzs[:, tzs.shape[1]//2])
                tcs = np.squeeze(tcs[:, tcs.shape[1]//2])
                writerX.add_embedding(tzs,
                                      metadata=tls,
                                      global_step=i,
                                      tag='zs')
                writerX.add_embedding(tcs,
                                      metadata=tls,
                                      global_step=i,
                                      tag='cs')
        except KeyboardInterrupt as e:
            print('Interrupted...')
            save(saver, sess, args.logdir, i)
            break
