import os


def delete_files_with_prefix(folder_path, prefix="._"):
    """
    Recursively deletes all files starting with a specified prefix in a given folder.

    :param folder_path: The path to the folder to search.
    :param prefix: The file prefix to look for (default is "._").
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith(prefix):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


# Example usage
folder_to_clean = "/home/datasets/UrbanBIS/"  # Replace with your folder path
delete_files_with_prefix(folder_to_clean)