import os
import re


def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory,
                                             followlinks=True):
        for filename in filenames:
            if re.search(pattern, filename):
                files.append(os.path.join(root, filename))
    return files