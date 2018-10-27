import librosa
import numpy as np
import os
import argparse
import sys
from utils import find_files
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root dir with audio files to process')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--n_mels', type=int, default=80,
                        help='Number of mel bands')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sample rate')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='Number of fft bins')
    parser.add_argument('--hop_size', type=int, default=660,
                        help='Hop size for FFT in samples')
    parser.add_argument('--clip_before_log', type=float, default=0.0001,
                        help='Value to clip small values in spectrogram'
                        ' before applying log')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=8)

    args = parser.parse_args()
    return args


def extract_and_save(f):
    try:
        au, sr = librosa.load(f, sr=args.sr)
        au = librosa.util.normalize(au)
        mel_bank = librosa.filters.mel(args.sr, args.n_fft, args.n_mels)
        spec = np.abs(librosa.stft(au, args.n_fft, args.hop_size).T)**2
        mel_spec = np.dot(mel_bank, spec.T).T
        log_spec = np.log(np.clip(spec, args.clip_before_log, None))
        log_mel = np.log(np.clip(mel_spec, args.clip_before_log, None))

        if not args.dry_run:
            file_folder = os.path.join(args.out_dir, *f[len(root_folder):].split(os.sep)[:-1])
            os.makedirs(file_folder, exist_ok=True)
            file_name = f.split(os.sep)[-1]
            np.savez(os.path.join(file_folder, file_name),
                     au=au,
                     log_spec=log_spec,
                     log_mel=log_mel)
    except Exception as e:
        print()
        print(e)
        print('File ' + f)


if __name__ == '__main__':
    args = get_args()
    files = find_files(args.data_dir, r'\.(wav|mp3|ogg|flac|au)')

    files = list(map(os.path.normpath, files))
    root_folder = os.path.commonpath(files)

    Parallel(n_jobs=args.n_jobs)(delayed(extract_and_save)(f) for f in tqdm(files))
    print('Done')
