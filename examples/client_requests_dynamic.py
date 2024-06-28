import json
from pprint import pprint
import requests

# Load some sample data
data = json.load(open("../data/spanish-news-seamless-one.json"))
one_sentence = data["data"][0]["INPUT_TEXT"][0]
print(f"Original Sentence:\n{one_sentence}")

base_url = "http://localhost:8000/v2/models"


### FastText Request
inference_request_fasttext = {
    "id": "fast_text_infer",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [one_sentence],
        }
    ],
}
result = requests.post(
    url=f"{base_url}/fasttext-language-identification/infer",
    json=inference_request_fasttext,
)
pprint(result.json())

### SeamlessM4Tv2Large Request
inference_request_seamless = {
    "id": "seamless",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [one_sentence],
        },
        {
            "name": "SRC_LANG",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": ["spa"],
        },
        {
            "name": "TGT_LANG",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": ["eng"],
        },
    ],
}
result = requests.post(
    url=f"{base_url}/seamless-m4t-v2-large/infer",
    json=inference_request_seamless,
)
pprint(result.json())
