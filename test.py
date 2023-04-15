from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
print(tokenizer('boot'))
print(tokenizer('boots'))

"""import random
import string

# Define the length of the string sequence
length = 10

# Define the pool of characters to choose from
characters = string.ascii_letters

# Generate the random string sequence
random_sequence = ''.join(random.choice(characters) for i in range(length))

# Print the random string sequence

origin = 'a [V] dog'
t = origin.replace('[V]', random_sequence)
print(t)"""