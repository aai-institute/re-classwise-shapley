"""
Hack, hack, hackety hack...
"""

import os
import re
import shutil


def find_image_paths(latex_file: str) -> list[str]:
    """
    Parses a LaTeX file and finds all paths used with \includegraphics.

    Args:
        latex_file: Path to the LaTeX file to parse.

    Returns:
         A list of image paths.
    """
    image_paths = []
    regex = r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^\}]+)\}"
    with open(latex_file, "r") as file:
        content = file.read()
        matches = re.findall(regex, content)
        image_paths.extend(matches)
    return image_paths


def copy_images_to_destination(image_paths: list[str], destination_root: str):
    """
    Copies image files to a new destination, preserving the file hierarchy.

    Args:
        image_paths: List of image file paths to copy.
        destination_root: The root directory to copy the files into.

    """
    for path in image_paths:
        source_path = os.path.normpath(path)
        if source_path.startswith("../"):
            destination_path = os.path.join(destination_root, source_path[3:])
        else:
            destination_path = os.path.join(destination_root, source_path)
        destination_dir = os.path.dirname(destination_path)

        os.makedirs(destination_dir, exist_ok=True)

        # Copy the file
        shutil.copy2(source_path, destination_path)
        print(f"Copied {source_path} to {destination_path}")


def replace_cite_with_citep(file_path: str):
    print(f"Replacing (\\cite) with \\citep in {file_path}")
    with open(file_path, 'r') as file:
        content = file.read()

    content = re.sub(r'\(\{\\cite\{(.*?)\}\}\)', r'{\\citep{\1}}', content)

    print(f"Replacing (\\citep) with \\citep in {file_path}")

    content = re.sub(r'\(\{\\citep\{(.*?)\}\}\)', r'{\\citep{\1}}', content)

    with open(file_path, 'w') as file:
        file.write(content)


if __name__ == "__main__":
    latex_file = "../report/tmlr-re-csshapley.tex"
    destination_root = "../report/"

    image_paths = find_image_paths(latex_file)

    copy_images_to_destination(image_paths, destination_root)

    replace_cite_with_citep(latex_file)
