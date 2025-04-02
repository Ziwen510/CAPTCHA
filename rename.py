import os

def replace_substring_in_filenames(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        if "__" in filename:
            # Create the new filename by replacing the substring
            new_filename = filename.replace("__", "_")
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    # Specify the directory where the files are located
    directory = "./checkpoints"  # Change to your target directory if needed
    replace_substring_in_filenames(directory)
