import os
import ast
import pkg_resources


def extract_imports_from_file(file_path):
    """Extracts imported libraries from a Python file."""
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)

    return imports


def generate_requirements_txt(directory):
    """
    Recursively searches through the directory for Python files and collects
    the libraries being imported, then generates a requirements.txt file.

    Args:
        directory (str): The directory to scan for Python files.
    """
    libraries = set()

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                libraries.update(extract_imports_from_file(file_path))

    # Get installed packages using pkg_resources
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    # Filter libraries to include only installed packages
    installed_libraries = libraries.intersection(installed_packages)

    # Write the results to requirements.txt
    with open('requirements.txt', 'w') as req_file:
        for lib in sorted(installed_libraries):
            req_file.write(f"{lib}\n")

    print("requirements.txt generated successfully.")


# Example usage
generate_requirements_txt('/path/to/your/project')
