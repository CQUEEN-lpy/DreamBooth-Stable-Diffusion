from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
print(tokenizer('a [V] dog'))
print(tokenizer('To generate a random string sequence in Python, you can use the random module and the string module. The string module provides a set of constants for different types of characters (such as digits, uppercase letters, lowercase letters, punctuation, etc.), and the random module provides functions for generating random numbers. Heres an example that generates a random string sequence of length 10:'))

import random
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
print(t)