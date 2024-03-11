def sort_and_clean_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Remove blank lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]

    # Function to extract numerical part for sorting
    def sort_key(line):
        parts = line.split()
        # Assuming the image name is the second part and is formatted like 'img123.bmp'
        image_name = parts[1]
        # Extract the number part from the image name (strip 'img' and '.bmp')
        number_part = int(image_name.replace('img', '').replace('.bmp', ''))
        return number_part

    # Sort lines by the extracted number part
    sorted_lines = sorted(lines, key=sort_key)

    with open(output_file, 'w') as f:
        for line in sorted_lines:
            f.write(f"{line}\n")


# Example usage
input_path = '../data/LIVE2/databaserelease2/gblur/info.txt'  # Update with the actual path
output_path = '../data/LIVE2/databaserelease2/gblur/info_sorted.txt'  # Path where the sorted file will be saved
sort_and_clean_file(input_path, output_path)
