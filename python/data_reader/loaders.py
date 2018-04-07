import numpy as np
import tensorflow as tf
import librosa
import os
import fnmatch
import random


def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory,
                                             followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


class BaseLoader(object):
    '''Base class for different data loaders'''

    def get_types(self):
        '''Should be overriden to return tuple of
        output types for use with Dataset API'''
        raise NotImplementedError

    def next(self):
        '''Produces new loader output each time it's called'''
        raise NotImplementedError


class WavToMelSpecLoader(BaseLoader):
    '''Loads data from wav files and extracts melspectrograms'''
    def __init__(self, root_dir, genres, sr, n_mel, n_fft, hop_size):
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
        self.root_dir = root_dir
        self.genres = genres
        self.sr = sr
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.files = find_files(self.root_dir, '*.au')
        self.all_genres = list(sorted(list(set([f.split(os.sep)[-2]
                                                for f in self.files]))))
        if len(genres) > 0:
            self.files = [f for f in self.files
                          if f.split(os.sep)[-2] in self.genres]

    def next(self):
        next_file = random.sample(self.files, 1)[0]
        au, _ = librosa.load(next_file, sr=self.sr)
        stftm = np.abs(librosa.stft(
            au,
            n_fft=self.n_fft,
            hop_length=self.hop_size))**2

        mel_filters = librosa.filters.mel(self.sr, self.n_fft, self.n_mel)
        mel_spec = np.dot(mel_filters, stftm)
        genre_id = self.all_genres.index(next_file.split(os.sep)[-2])

        return (mel_spec.T, genre_id)

    def get_types(self):
        return (tf.float32, tf.float32)

    def get_shapes(self):
        return ([None, self.n_mel], [])
