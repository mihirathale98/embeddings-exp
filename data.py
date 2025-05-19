import json
from tqdm import tqdm

def load_json(file_path):
    """
    Load a JSON file and return its content.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


data = load_json('hotpot_test_fullwiki_v1.json')


all_questions = [d['question'] for d in data]
all_contexts = [d['context'] for d in data]
context_paras = {}
for context in tqdm(all_contexts):
    for cont in context:
        para = ' '.join(cont[1])
        title = cont[0]
        context_paras[title] = para


def save_json(data, file_path):
    """
    Save data to a JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


## Save the context paragraphs to a JSON file
save_json(context_paras, 'context_paras.json')
## Save the questions to a JSON file
save_json(all_questions, 'questions.json')