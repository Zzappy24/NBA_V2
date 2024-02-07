import os
import ast

def find_libraries_in_directory(directory):
    libraries = set()

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=file_path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    libraries.add(alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                libraries.add(node.module)
                    except SyntaxError as e:
                        print(f"SyntaxError in {file_path}: {e}")

    return libraries

if __name__ == "__main__":
    target_directory = "./"
    used_libraries = find_libraries_in_directory(target_directory)

    print("Bibliothèques utilisées dans le projet :")
    for lib in used_libraries:
        print(lib)
