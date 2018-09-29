# script for recording gtzan with mic for some kind of data augmentation
# recorded data is what you'll probably need to deal as well after all
import librosa
import sounddevice as sd
import soundfile as sf
import numpy as np
import argparse
import os
from utils import find_files
import sys
import resampy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--out_sr', type=int, default=16000)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    files = find_files(args.data_dir, '*.au')

    files = list(map(os.path.normpath, files))
    root_folder = os.path.commonpath(files)

    for i, f in enumerate(files):
        au, sr = librosa.load(f, sr=None)
        au = librosa.util.normalize(au)

        sys.stdout.write('File {} out of {} \r'.format(i, len(files)))
        sys.stdout.flush()

        recorded = sd.playrec(au, samplerate=sr, blocking=True, channels=1)
        recorded = resampy.resample(recorded[:, 0], sr, args.out_sr)
        file_folder = os.path.join(args.out_dir, *f[len(root_folder):].split(os.sep)[:-1])
        os.makedirs(file_folder, exist_ok=True)
        file_name = f.split(os.sep)[-1]
        sf.write(os.path.join(file_folder, file_name + '.ogg'), recorded, args.out_sr)
