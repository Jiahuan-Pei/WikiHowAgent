import os
import sys
import glob

def print_txt_files(directory, suffix='txt'):
    # Construct the search pattern
    pattern = os.path.join(directory, f'**/*.{suffix}')
    
    # Find all .txt files in the directory
    txt_files = glob.glob(pattern, recursive=True)
    print('\n'.join(txt_files))
    print(f'Total number of doc end with .{suffix}:', len(txt_files))

# Example usage
# directory_path = '/path/to/your/directory'  # Replace with your directory path
print_txt_files(directory=sys.argv[1])