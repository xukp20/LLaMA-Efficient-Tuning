input_path = "/cephfs/xukangping/code/LLaMA-Efficient-Tuning/data/metamathQA.json"
output_path = "/cephfs/xukangping/code/LLaMA-Efficient-Tuning/data/metamathQA_parsed.json"
# in: {query, response}
# out {instruction=query, input="", output=response}

import json
from tqdm import tqdm
with open(input_path, "r") as f:
    data = json.load(f)

outputs = []
for item in tqdm(data):
    outputs.append({
        "instruction": item["query"],
        "input": "",
        "output": item["response"]
    })

with open(output_path, "w") as f:
    for item in outputs:
        f.write(json.dumps(item) + "\n")