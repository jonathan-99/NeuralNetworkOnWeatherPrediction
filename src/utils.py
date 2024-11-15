"""
common utility
"""

# utils.py

import os
import ast


def extract_imports_from_file(file_path):
    """
    Extract the import libraries from a Python file.

    Args:
        file_path (str): Path to the Python file to scan.

    Returns:
        set: A set of unique libraries imported in the file.
    """
    imports = set()

    with open(file_path, 'r') as file:
        # Parse the Python file to get the abstract syntax tree (AST)
        tree = ast.parse(file.read())

        # Extract import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)

    return imports


def get_all_python_files(directory):
    """
    Get a list of all Python files in the given directory and subdirectories.

    Args:
        directory (str): Directory to search for Python files.

    Returns:
        list: List of Python file paths.
    """
    python_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files


def generate_requirements_txt(directory, output_file='requirements.txt'):
    """
    Generate a requirements.txt file with all import libraries from Python files in the given directory.

    Args:
        directory (str): Directory to scan for Python files.
        output_file (str): Name of the output requirements file (default: 'requirements.txt').
    """
    all_imports = set()

    # Get all Python files in the directory and subdirectories
    python_files = get_all_python_files(directory)

    # Extract imports from each file
    for python_file in python_files:
        imports = extract_imports_from_file(python_file)
        all_imports.update(imports)

    # Write the requirements.txt file
    with open(output_file, 'w') as req_file:
        for imp in sorted(all_imports):
            req_file.write(f"{imp}\n")

    print(f"Requirements have been written to {output_file}")


# Example usage
if __name__ == "__main__":
    # Set the directory to scan (e.g., current directory or specific folder)
    directory_to_scan = '.'
    generate_requirements_txt(directory_to_scan)
