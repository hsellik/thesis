import errno
import glob
import math
import os
import shutil

# Source of the original projects/directory to extract Java files from
SOURCE_DIR = "/Users/hendrig/workspace/bug-detection/code2vec/data/testing/"
# Temp directory to hold cloned java files
TEMP_DIR = "/Users/hendrig/workspace/temp/"
# Directories to store resulting training/testing/validation files
TRAINING_DIR = "/Users/hendrig/workspace/java_dataset/training/"
TESTING_DIR = "/Users/hendrig/workspace/java_dataset/testing/"
VALIDATION_DIR = "/Users/hendrig/workspace/java_dataset/validation/"

def clone_java_files(src, dest):
    if not os.path.exists(dest):
        try:
            os.makedirs(os.path.dirname(dest))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    for file_path in glob.glob(os.path.join(src, '**', '*.java'), recursive=True):
        new_path = os.path.join(dest, os.path.basename(file_path))
        shutil.copy(file_path, new_path)


def copy_single_file(src, dest):
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        try:
            os.makedirs(os.path.dirname(dest))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    shutil.copy(src, dest)


def create_dataset():
    clone_java_files(SOURCE_DIR, TEMP_DIR)

    total_count = sum([len(files) for r, d, files in os.walk(TEMP_DIR)])
    training_count = math.floor(total_count * 0.75)
    testing_count = math.floor(total_count * 0.2)
    # validation_count not needed as it's just the rest of the files

    for _, _, files in os.walk(TEMP_DIR):
        for filename in files[0:training_count - 1]:
            source = os.path.join(TEMP_DIR, filename)
            destination = os.path.join(TRAINING_DIR, filename)
            copy_single_file(source, destination)

        for filename in files[training_count: (training_count + testing_count - 1)]:
            source = os.path.join(TEMP_DIR, filename)
            destination = os.path.join(TESTING_DIR, filename)
            copy_single_file(source, destination)

        for filename in files[training_count + testing_count: -1]:
            source = os.path.join(TEMP_DIR, filename)
            destination = os.path.join(VALIDATION_DIR, filename)
            copy_single_file(source, destination)
        break

if __name__ == '__main__':
    create_dataset()