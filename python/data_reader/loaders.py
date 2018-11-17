import numpy as np
import tensorflow as tf
import os
import random
import pandas as pd
import h5py


def get_loader(name):
    return globals()[name]


class BaseLoader(object):
    '''Base class for different data loaders'''
    def __init__(self,
                 folder,
                 **kwargs):
            ''' Args:
                folder -> str - folder with train.txt and test.txt,
                    which list training and testing file locations'''
            self.train_files = []
            with open(os.path.join(folder, 'train.txt')) as f:
                for l in f:
                    if l:
                        self.train_files.append(l.rstrip())
            self.test_files = []
            with open(os.path.join(folder, 'test.txt')) as f:
                for l in f:
                    if l:
                        self.test_files.append(l.rstrip())
            self.files = self.train_files + self.test_files

    def get_types(self):
        '''Should be overriden to return tuple of
        output types for use with Dataset API'''
        raise NotImplementedError

    def load(self, file, *args, **kwargs):
        '''Loads and preprocesses given file and arguments'''
        raise NotImplementedError

    def next_train(self):
        '''Samples and loads files for train batches'''
        return self.load(random.sample(self.train_files, 1)[0])

    def next_test(self):
        '''Samples and loads files for test batches '''
        return self.load(random.sample(self.test_files, 1)[0])


class H5FeatureLoader(BaseLoader):
    def __init__(self,
                 folder,
                 feature_type,
                 crop_size=None,
                 fma_meta_path=None,
                 gtzan_labels=False,
                 **kwargs):
        ''' Args:
            folder -> str - arg for BaseLoader
            feature_type -> str - what feature to extract from h5 file
            crop_size -> None or int - how much of input to randomly crop
            fma_meta_path -> str - folder path of unarchived FMA dataset
                metadata for adding main genre to embeddings summary
            gtzan_labels -> bool - whether to use GTZAN labels for
                classification
        '''
        self.feature_type = feature_type
        self.fma_meta_path = fma_meta_path
        self.fma_meta = None
        self.crop_size = crop_size
        if self.fma_meta_path is not None:
            self.fma_meta = pd.read_csv(
                os.path.join(self.fma_meta_path, 'tracks.csv'),
                header=[0, 1],
                index_col=0)['track']
        self.gtzan_labels = gtzan_labels
        self.all_genres = ['blues', 'classical', 'country', 'disco',
                           'hiphop', 'jazz', 'metal', 'pop', 'reggae',
                           'rock']
        return super().__init__(folder)

    def get_types(self):
        if not self.gtzan_labels:
            return {'path': tf.string,
                    'clean_batch': tf.float32,
                    'metadata': tf.string}
        else:
            return {'path': tf.string,
                    'clean_batch': tf.float32,
                    'metadata': tf.string,
                    'label': tf.float32}

    def get_shapes(self):
        sample_batch = self.load(self.train_files[0])
        self.feat_shape = [
            *sample_batch['clean_batch'].shape]
        self.feat_shape[0] = None

        if self.gtzan_labels:
            self.label_shape = sample_batch['label'].shape
            return {'path': [],
                    'clean_batch': self.feat_shape,
                    'metadata': [],
                    'label': self.label_shape}
        else:
            return {'path': [],
                    'clean_batch': self.feat_shape,
                    'metadata': []}

    def load(self, path):
        with h5py.File(path, 'r') as f:
            if (self.crop_size is None or
                    self.crop_size > f[self.feature_type].shape[0]):
                feat = f[self.feature_type][:]
            else:
                start = random.randint(
                    0, f[self.feature_type].shape[0] - self.crop_size)
                feat = f[self.feature_type][start: start + self.crop_size]
        if self.feature_type == 'au':
            feat = feat[:, None]
        genre = 'N/A'
        if self.fma_meta is not None:
            track_id = path.split(os.sep)[-1].split('.')[0]
            try:
                genre = self.fma_meta.loc[int(track_id)]['genre_top']
            except Exception as e:
                genre = 'N/A'
        if self.gtzan_labels:
            genre = path.split(os.sep)[-2]
            genre_id = self.all_genres.index(genre)
            one_hot = np.zeros(len(self.all_genres))
            one_hot[genre_id] = 1
            return {'path': path,
                    'clean_batch': feat,
                    'metadata': genre,
                    'label': one_hot}
        return {'path': path,
                'clean_batch': feat,
                'metadata': genre}
