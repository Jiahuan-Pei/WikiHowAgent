import os
import sys
import glob

def delete_txt_files(directory='data/wikihow', suffix='txt'):
    # Construct the search pattern
    pattern = os.path.join(directory, f'**/*.{suffix}')
    
    # Find all .txt files in the directory
    txt_files = glob.glob(pattern, recursive=True)
    
    # Loop through the list of file paths & remove each file
    for txt_file in txt_files:
        try:
            os.remove(txt_file)
            print(f"Deleted: {txt_file}")
        except Exception as e:
            print(f"Error deleting {txt_file}: {e}")

# Example usage
# directory_path = '/path/to/your/directory'  # Replace with your directory path
delete_txt_files(directory=sys.argv[1])