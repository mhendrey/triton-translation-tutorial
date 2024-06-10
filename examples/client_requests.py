import json
from pprint import pprint
import requests

# Example data with just one article in it
news_articles = json.load(open("../data/spanish-news-one.json"))
# Grab the text of the first document. document is a long text string
document = news_articles["data"][0]["INPUT_TEXT"][0]

# Grab chunks for the seamless endpoint. Chunks is a list of strings
data = json.load(open("../data/spanish-news-seamless-one.json"))
chunks_data = data["data"][0]["INPUT_TEXT"]
chunks = chunks_data["content"]

base_url = "http://localhost:8000/v2/models"

### Translate Requests
## Default Translate
inference_request = {
    "id": "inferece_no_optional_params",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": [document],
        }
    ],
}
result = requests.post(url=f"{base_url}/translate/infer", json=inference_request)
print("Translate Request with no optional params")
pprint(result.json())

## Translate with optional src_lang
inference_request_src_lang = {
    "id": "inference_with_src_lang",
    "parameters": {"src_lang": "spa"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": [document],
        }
    ],
}
result = requests.post(
    url=f"{base_url}/translate/infer", json=inference_request_src_lang
)
print("Translation, src_lang provided")
pprint(result.json())

## Translate with optional src_lang & tgt_lang
inference_request_src_lang = {
    "id": "inference_with_src_lang",
    "parameters": {"src_lang": "spa", "tgt_lang": "fra"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": [document],
        }
    ],
}
result = requests.post(
    url=f"{base_url}/translate/infer", json=inference_request_src_lang
)
print("Translation, src_lang provided & tgt_lang is French")
pprint(result.json())

### FastText Request
inference_request_fasttext = {
    "id": "fast_text_infer",
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": [document],
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
            "shape": [len(chunks)],
            "datatype": "BYTES",
            "data": chunks,
        },
        {
            "name": "SRC_LANG",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["spa"],
        },
        {
            "name": "TGT_LANG",
            "shape": [1],
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
