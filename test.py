import re

my_string = 'xxxx  xxxx  xxxx          xxx'
my_string = re.sub(r'\s+', ' ', my_string)
print(my_string)