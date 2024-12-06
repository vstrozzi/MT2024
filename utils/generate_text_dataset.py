# Set your API key
from imagenet_classes import imagenet_classes
from openai import OpenAI

api_key = "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"
client = OpenAI(
    api_key=api_key,  
)

input_file = "top_1500_nouns_clean.txt"

# Read the input file and remove duplicates
with open(input_file, "r") as file:
    lines = [line.strip() for line in file]

def generate_descriptions(class_name):
    """Generate 10 basic descriptions for a given class."""
    prompt = (
        f'''Generate 5 very simple descriptions for the class '{class_name}', tailored for a CLIP model. Each description should aim to capture the concept as visually as possible. If the concept can be divided into meaningful subparts, provide one description for each subpart to maximize coverage. The final description should clearly summarize the concept in a short and simple way. For example:

For "shark":
An image of a large fish.
An image of a gray animal.
An image of a shark swimming.
An image of a predator underwater.
An image of a shark.

For "banana":
An image of a yellow fruit.
An image of a curved object.
An image of a banana on a table.
An image of a peeled banana.
An image of a banana.

For "seasons":
An image of green trees in summer.
An image of red and orange leaves in autumn.
An image of snow on the ground in winter.
An image of flowers blooming in spring.
An image of the four seasons.

For "freedom":
An image of a bird flying in the sky.
An image of a person with open arms.
An image of an empty road.
An image of a waving flag.
An image of freedom.

Descriptions should prioritize clear, visual traits or representations. When dealing with abstract concepts, break them into smaller, clear components and provide visual or textual representations. Output only the descriptions as plain text, separated by line breaks (\n).'''
    )
    
    response =  client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# Create and save descriptions in a text file
output_file = "top_1500_nouns_5_sentences.txt"
print(lines)
with open(output_file, "a") as file:
    for i, class_name in enumerate(lines):
        print(class_name)
        if i <= 1064:
            continue
        try:
            descriptions = generate_descriptions(class_name)
            file.write(f"{descriptions}\n")
        except Exception as e:
            continue
        print(f"Currently at iteration {i}")
print(f"Descriptions have been saved to {output_file}.")

