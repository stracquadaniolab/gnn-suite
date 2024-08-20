#!/usr/bin/env python3
import sys
import os
import typer


def split_file(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    notes_header = lines[0]
    header = lines[1].lstrip()

    train_lines = [line.lstrip("Train:").lstrip() for line in lines if line.startswith('Train:')]
    test_lines = [line.lstrip("Test:").lstrip() for line in lines if line.startswith('Test:')]
    all_lines = [line.lstrip("All:").lstrip() for line in lines if line.startswith('All:')]

    basename = os.path.splitext(input_file)[0]
    with open(basename + '-train.txt', 'w') as f:
        f.write(notes_header)
        f.write(header)
        f.writelines(train_lines)
    with open(basename + '-test.txt', 'w') as f:
        f.write(notes_header)
        f.write(header)
        f.writelines(test_lines)
    with open(basename + '-all.txt', 'w') as f:
        f.write(notes_header)
        f.write(header)
        f.writelines(all_lines)

if __name__ == "__main__":
    split_file(sys.argv[1])
