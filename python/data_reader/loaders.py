import numpy as np
import tensorflow as tf
import librosa
import os
import sys
import fnmatch
import random

sys.path.append(os.path.join(sys.path[1], os.path.abspath('../')))

from utils import find_files


def get_loader(name):
    return globals()[name]


class BaseLoader(object):
    '''Base class for different data loaders'''
    def __init__(self,
                 folder,
                 file_ext,
                 test_size=0.2,
                 random_seed=42,
                 **kwargs):
            self.files = find_files(folder, file_ext)
            random.seed(random_seed)
            indx = list(range(len(self.files)))
            random.shuffle(indx)
            train_size = int(len(self.files) * (1 - test_size))
            self.train_files = list(np.array(self.files)[indx[:train_size]])
            self.test_files = list(np.array(self.files)[indx[train_size:]])

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


class WavToLogMelLoader(BaseLoader):
    '''Loads data from wav files and extracts melspectrograms'''
    def __init__(self,
                 root_dir,
                 genres,
                 sr, n_mel,
                 n_fft,
                 hop_size,
                 test_size=0.2,
                 random_seed=None):
        ''' Args:
            root_dir -> str - directory to scan for .au files
                (GTZAN dataset file format)
            genres -> list of str - If specified,
                which genres folders should be present in data,
                all genres are used if list is empty
            sr -> int - audio sample rate
            n_mel -> int - number of mel filterbank
            n_fft -> int - number of fft bins
            hop_size -> int - fft hop_size'''
        super().__init__(root_dir, r'\.(au|wav|ogg|mp3)',
                         test_size, random_seed)
        self.root_dir = root_dir
        self.genres = genres
        self.sr = sr
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        # precomputed filters
        self.mel_filters = librosa.filters.mel(self.sr, self.n_fft, self.n_mel)
        self.all_genres = list(sorted(list(set([f.split(os.sep)[-2]
                                                for f in self.files]))))
        if genres is not None and len(genres) > 0:
            self.files = [f for f in self.files
                          if f.split(os.sep)[-2] in self.genres]

    def get_types(self):
        return (tf.float32, tf.float32)

    def get_shapes(self):
        return ([None, self.n_mel], [])

    def load(self, path):
        au, _ = librosa.load(path, sr=self.sr)
        au = librosa.util.normalize(au)
        stftm = np.abs(librosa.stft(
            au,
            n_fft=self.n_fft,
            hop_length=self.hop_size))**2

        mel_spec = np.dot(self.mel_filters, stftm)
        mel_log_spec = np.log(np.clip(mel_spec, 0.0001, None))
        genre_id = self.all_genres.index(path.split(os.sep)[-2])

        return (mel_spec.T.astype(np.float32), int(genre_id))


class PrecomputedMelSpecLoader(BaseLoader):
    def __init__(self,
                 root_dir,
                 genres=None,
                 recorded_feat_root_folder=None,
                 sr=22050,
                 n_mel=80,
                 n_fft=2048,
                 hop_size=660,
                 test_size=0.2,
                 random_seed=None):
        ''' Args:
            root_dir -> str - directory to scan for .au files
                (GTZAN dataset file format)
            genres -> list of str - If specified,
                which genres folders should be present in data,
                all genres are used if list is empty
            sr -> int - audio sample rate
            n_mel -> int - number of mel filterbank
            n_fft -> int - number of fft bins
            hop_size -> int - fft hop_size'''
        super().__init__(root_dir, r'\.(au.npz|wav.npz|ogg.npz|mp3.npz)',
                         test_size, random_seed)
        self.recorded_feat_root_folder = recorded_feat_root_folder
        if self.recorded_feat_root_folder is not None:
            self.additional_train_files = [os.path.join(
                recorded_feat_root_folder,
                *f.split(os.sep)[-2:])[:-7] + '.au.ogg.npz' for f in self.train_files]
            self.additional_test_files = [os.path.join(
                recorded_feat_root_folder,
                *f.split(os.sep)[-2:])[:-7] + '.au.ogg.npz' for f in self.test_files]
            #import ipdb; ipdb.set_trace()
            self.train_files.extend(self.additional_train_files)
            #self.test_files.extend(self.additional_test_files)
        self.root_dir = root_dir
        self.genres = genres
        self.sr = sr
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.all_genres = list(sorted(list(set([f.split(os.sep)[-2]
                                                for f in self.files]))))
        if genres is not None and len(genres) > 0:
            self.files = [f for f in self.files
                          if f.split(os.sep)[-2] in self.genres]
        if genres is None:
            self.genres = self.all_genres

    def get_types(self):
        return (tf.float32, tf.float32)

    def get_shapes(self):
        return ([None, self.n_mel], [len(self.genres)])

    def load(self, path):
        mel_spec = np.load(path)['log_mel']
        genre_id = self.all_genres.index(path.split(os.sep)[-2])
        one_hot = np.zeros(len(self.genres))
        one_hot[genre_id] = 1
        return (mel_spec.astype(np.float32),
                one_hot)
