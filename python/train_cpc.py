import os
import numpy as np
import tensorflow as tf
# I guess it's just easier for projections
# than plain tensorboard
from tensorboardX import SummaryWriter
import argparse
from models import get_model
from data_reader import create_data_reader, get_loader
from utils import load, save
import json
import time


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
    parser.add_argument('--n_steps', type=int, default=int(1e6))
    parser.add_argument('--n_losses', type=int, default=20,
                        help='Number of future-predicting losses')
    parser.add_argument('--crop_size', type=int, default=150,
                        help='Random crop inputs to this amount of frames')
    parser.add_argument('--l2_coeff', type=float, default=0,
                        help='L2 regularization for zs and cs embeddings '
                             'regularization')
    return parser.parse_args()


def get_f_predictions(zs, cs, k, n_units=1):
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

        prediction_layer = tf.layers.Dense(
            n_units,
            activation=tf.exp,
            name='f_prediction_layer_' + str(k))

        f_pred = prediction_layer(zs_cs_concat)

    with tf.variable_scope('contrastive_example_' + str(k)):
        shuffled_zs = tf.concat((shifted_zs[1:, :, :],
                                 shifted_zs[:1, :, :]),
                                0)
        shuffled_zs = tf.stop_gradient(shuffled_zs)
        contr_concat = tf.stop_gradient(tf.concat(
            (shuffled_zs, shifted_cs), 2))
        contrastive_f_pred = prediction_layer(contr_concat)
    with tf.variable_scope('accuracy_' + str(k)):
        acc = (tf.reduce_sum(
                tf.where(
                    tf.greater(f_pred, contrastive_f_pred),
                    tf.ones_like(f_pred, dtype=tf.float32),
                    tf.zeros_like(f_pred, dtype=tf.float32))) /
               tf.reduce_prod(tf.cast(tf.shape(zs), tf.float32)))

    return f_pred, contrastive_f_pred, acc


def crop_inputs(im, size):
    return tf.random_crop(im, [size, tf.shape(im)[-1]])

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

    if args.crop_size > 0:
        def crop_fn(im, *args_):
            return (crop_inputs(im, args.crop_size), *args_)
        (next_element, labels), train_data_init_op, test_data_init_op = (
            create_data_reader(loader, args.batch_size, args.test_batch_size,
                               [crop_fn]))
    else:
        (next_element, labels), train_data_init_op, test_data_init_op = (
            create_data_reader(loader, args.batch_size, args.test_batch_size))

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
            loss = -tf.reduce_mean(
                tf.log(f_pred/tf.reduce_sum(contrastive_f_pred, axis=0)))

            pred_shape = tf.shape(f_pred)
            batch = tf.cast(pred_shape[0], tf.float32)
            time_steps = tf.cast(pred_shape[1], tf.float32)
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

    for i in range(ckpt_step, args.n_steps):
        try:
            step_start = time.time()
            loss_, mean_MI_, sums, zs, cs, ls, _ = sess.run([
                total_loss,
                mean_MI,
                sums_op,
                model.zs,
                model.cts,
                labels,
                opt])
            step_time = time.time() - step_start
            print('Step:{}. Elapsed:{:.3f}. Loss: {:.3f}, MI: {:.3f}'
                  .format(i, step_time, loss_, mean_MI_))
            if i % args.test_every == 0:
                sess.run(test_data_init_op)
                loss_, mean_MI_, sums, tzs, tcs, tls = sess.run([
                    total_loss,
                    mean_MI,
                    test_sums_op,
                    model.zs,
                    model.cts,
                    labels])
                print('Test loss: {:3f}, test MI: {:3f}'.
                      format(loss_, mean_MI_))
                writer.add_summary(sums, i)
                sess.run(train_data_init_op)
            if i % args.save_every == 0:
                save(saver, sess, args.logdir, i)
            if i % args.summaries_every:
                writer.add_summary(sums, i)
            if i % args.embeddings_every == 0:
                # not sure if it's ok to take mean
                print('Saving projections')
                tzs = np.squeeze(np.mean(tzs, axis=1))
                tcs = np.squeeze(np.mean(tcs, axis=1))
                tls = [loader.genres[np.argmax(l)] for l in tls]
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
