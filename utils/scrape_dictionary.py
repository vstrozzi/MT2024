import requests
from bs4 import BeautifulSoup

# URL of the webpage containing the list of nouns
url = 'https://www.talkenglish.com/vocabulary/top-1500-nouns.aspx'

# Input and output file paths
input_file_path = "top_1500_nouns.txt"  # Replace with your input file name
output_file_path = "top_1500_nouns_clean.txt"  # Replace with your desired output file name

# Process the file
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        # Extract the word (before the first tab or space)
        word = line.split()[0]
        # Write the word to the output file
        outfile.write(word + '\n')

print(f"Words have been written to {output_file_path}.")
