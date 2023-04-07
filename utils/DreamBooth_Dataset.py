from torch.utils import data

class DreamBooth_Dataset(data.Dataset):
    def __init__(self, subject, generated_method, real_prompt_json, generated_prompt_json):
        self.subject = subject
        self.generated_method = generated_method
        self.real_prompt_json= real_prompt_json
        self. generated_prompt_json = generated_prompt_json


