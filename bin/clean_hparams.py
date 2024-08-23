#!/usr/bin/env python3
import sys
import os
import re  # re module is needed for the regular expression
import typer


def clean_file(input_file):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Preserve the first two lines
    header = lines[:2]

    # Find the "Best trial:" line to capture the final block of information
    best_trial_index = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Best trial:"):
            best_trial_index = i
            break

    # Capture the last block containing the best trial information
    footer = lines[best_trial_index:]

    # Remove any unwanted lines that contain strange characters
    cleaned_footer = []
    for line in footer:
        cleaned_line = re.sub(r'[^\x20-\x7E]', '', line)  # Remove non-printable characters
        if cleaned_line.strip():  # Ensure the line is not empty after cleaning
            cleaned_footer.append(cleaned_line + '\n')

    # Write the cleaned content back to a new file
    basename = os.path.splitext(input_file)[0]
    with open(basename + ".txt", 'w', encoding='utf-8') as f:
        f.writelines(header)
        f.writelines(cleaned_footer)

    #print(f"Cleaned file written to {basename}-cleaned.txt")


if __name__ == "__main__":
    clean_file(sys.argv[1])
