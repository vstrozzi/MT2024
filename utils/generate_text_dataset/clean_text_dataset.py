### Clean file File paths
input_file = "top_1500_nouns_5_sentences.txt"  # Replace with your input file name
output_file = "top_1500_nouns_5_sentences_imagenet_clean.txt"  # Replace with your desired output file name

# Read the input file and remove duplicates
with open(input_file, "r") as file:
    lines = [line.strip() for line in file  if line.strip()]

# Strip whitespace and remove duplicates
unique_lines = list(set(line.strip() for line in lines))

# Sort unique lines (optional, for better readability)
unique_lines.sort()

# Write the unique lines to the output file
with open(output_file, "w") as file:
    file.write("\n".join(unique_lines) + "\n")

# Print the lengths and duplicate count
original_count = len(lines)
unique_count = len(unique_lines)
duplicate_count = original_count - unique_count

print(f"Original number of lines: {original_count}")
print(f"Number of unique lines: {unique_count}")
print(f"Number of duplicates removed: {duplicate_count}")
