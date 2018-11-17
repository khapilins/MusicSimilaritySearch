import numpy as np
import tensorflow as tf
import argparse
from models import get_model
from data_reader import create_data_reader, get_loader
from utils import load, save
from tensorboardX import SummaryWriter
import json
import time
import os


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
    parser.add_argument('--summaries_every', type=int, default=100,
                        help='Writes summaries on disk every summaries_every')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=int(1e5) + 1)
    parser.add_argument('--embeddings_every', type=int, default=10000,
                        help='Writes embeddings projections on disk every '
                        'embeddings_every')
    return parser.parse_args()

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

    next_element = reader.get('aug_batch', reader['clean_batch'])
    labels = reader['label']
    metadata = reader['metadata']

    next_element = tf.expand_dims(next_element, axis=2)

    model_out = model.create_model(next_element)

    writer = tf.summary.FileWriter(args.logdir,
                                   tf.get_default_graph())

    writerX = SummaryWriter(os.path.join(args.logdir, 'projections'))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=labels,
                logits=model_out))
        loss_sum = tf.summary.scalar('loss', loss)
        test_loss_sum = tf.summary.scalar('test_loss', loss)
    with tf.name_scope('acc'):
        acc = tf.reduce_sum(
            tf.cast(
                tf.equal(tf.argmax(labels, 1),
                         tf.argmax(model_out, 1)),
                tf.float32)) / tf.cast(tf.shape(labels)[0],
                                       tf.float32)
        acc_sum = tf.summary.scalar('accuracy', acc)
        test_acc_sum = tf.summary.scalar('test_accuracy', acc)

    sums_op = tf.summary.merge([acc_sum, loss_sum])
    test_sums_op = tf.summary.merge([test_acc_sum, test_loss_sum])
    optimizer = tf.train.AdamOptimizer(args.lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(train_data_init_op)
    saver = tf.train.Saver()
    ckpt_step = load(saver, sess, args.logdir, args.ckpt)

    for i in range(ckpt_step, args.n_steps):
        try:
            step_start = time.time()
            loss_, acc_, sums, meta_, emb_, _ = sess.run(
                [loss, acc, sums_op, metadata, model.dense_out, opt])
            step_time = time.time() - step_start
            print('Step:{}. Elapsed:{:.3f}. Loss: {:.3f}, accuracy: {:.3f}'
                  .format(i, step_time, loss_, acc_))
            if i % args.test_every == 0:
                sess.run(test_data_init_op)
                loss_, acc_, sums, meta_, emb_ = sess.run(
                    [loss, acc, test_sums_op, metadata, model.dense_out])
                print('Test loss: {:3f}, test accuracy: {:3f}'.
                      format(loss_, acc_))
                writer.add_summary(sums, i)
                sess.run(train_data_init_op)
            if i % args.save_every == 0:
                save(saver, sess, args.logdir, i)
            if i % args.summaries_every == 0:
                writer.add_summary(sums, i)
            if i % args.embeddings_every == 0:
                print('Saving projections')
                emb_ = np.squeeze(emb_[:, emb_.shape[1]//2])
                writerX.add_embedding(emb_,
                                      metadata=meta_,
                                      global_step=i,
                                      tag='dense_out')
        except KeyboardInterrupt as e:
            print('Interrupted...')
            save(saver, sess, args.logdir, i)
            break
