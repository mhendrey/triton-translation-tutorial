# Tutorial 3: Business Scripting Logic (BLS)
In the previous tutorials, we worked on the basic models that will be needed and added
dynamic batching to improve their throughput. We could stop here, but that would force
our requestors to do two things

1. Chunk their documents into the correct size before sending each chunk for
   translation
2. Submit each chunk in an asynchronous way. Otherwise, if they simply loop through
   and send each chunk sequential, they will see very slow translation times

This seems like a rather onerous burden upon the requestors. To make things simpler for
them, we will implement a another Triton deployment that leverages the Business
Scripting Logic
([BLS](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#business-logic-scripting))
to handle the document chunking and submit those chunks to the models we deployed in
Tutorial 2. With this, requestors will be able to submit entire documents for
translation and get the entire translated document back.

This Triton deployment will likely be the primary way for our requestors to interact
with our machine translation server. Of course, more advanced users could still decide
to create their own custom document chunking strategies and leverage just the
seamless-m4t-v2-large and/or fasttext-language-identification deployments if they wish.
But this will likely be the exception and not the rule, so we want to make this
deployment as requestor friendly as possible.

Triton's BLS enables you to have branching logic combined with submitting requests to
other models already deployed. By default, this BLS deployment will submit each chunk
for language identification before submitting it for translation. However, if the
requestor already knows the source language, then they can provide that as an optional
request parameter and skip this step.

Let's call this service level deployment `translate`. For this initial version, we will
use a basic chunking strategy of splitting on '.' Obviously, this is rather simplistic,
but it let's us get started. Just like the previous deployments, you can find the
needed code in the `model_repository` under the `translate` directory.

The BLS is supported by the Python backend. So just like in our previous Triton
deployments from the earlier tutorials, we need to specify both a config.pbtxt and a
Python file that contains the necessary code.

## config.pbtxt
Here is the entire contents of the file. We give our deployment a nice easy name for our
requestors to know. The BLS is supported by the Python backend. We will start off with
dynamic batching disabled, just like before. Our next tutorial will be a version that has
dynamic batching enabled. The input is expecting an array with a single string which is
the contents of the document to be translated. The output will the same, but now with
the translated text. Because this deployment will submit requests to the
seamless-m4t-v2-large deployment, it won't need to run on a GPU.

```
name: "translate"
backend: "python"
max_batch_size: 0
default_model_filename: "translate.py"

input [
  {
    name: "INPUT_TEXT"
    data_type: TYPE_STRING
    dims: [1]
  }
]
output [
  {
    name: "TRANSLATED_TEXT"
    data_type: TYPE_STRING
    dims: [1]
  }
]

instance_group [{ kind: KIND_CPU }]
```
## Python Backend
Again, the key function to implement is the `execute(requests)` class method in the
`TritonPythonModel`. A key difference is that this will be an `async` method. We will
loop through the list of translation requests and return a list of responses. Here are
the major steps we will take:

1. Iterate through each request in the input list.
2. For each request, extract the input text and optional parameters (source and target
   languages).
3. If the source language isn't provided, submit a request to the FastText model to
   identify the language.
4. The input text is split into chunks using `chunk_document()`.
5. Each chunk is sent to a SeamlessM4Tv2Large model for translation using asynchronous
   requests.
6. The translated chunks are collected and combined into a single document using
    `combine_translated_chunks()`.
7. The final translated document is wrapped in a Triton inference response and added to
   the list of responses to be returned.

This structure allows for efficient processing of large documents by breaking them into
manageable pieces and leveraging asynchronous operations for improved performance.
However, since we haven't done dynamic batching yet, we will only process a single
document at a time. This will likely affect performance just as before, but at least
a single document will be handled efficiently.

### New Code Features
The first bit of code that is different from previous tutorials demonstrates how to get
optional request parameters that may be passed by your requestors. 

```
# Get any optional parameters passed in.
request_params = json.loads(request.parameters())
src_lang = request_params.get("src_lang", None)
tgt_lang = request_params.get("tgt_lang", "eng")
```

Here we see that requestors can provide either the source language of their document,
`src_lang` or they can specify the target language, `tgt_lang`. If they don't provide
the `src_lang` each chunk will be submitted to FastText for identification. If they
don't provde a `tgt_lang` it defaults to English.

MORE TEXT NEEDS TO GO HERE


## Performance Analyzer
Just like before, we can leverage the built in perf_analyzer available in the Triton
Inference Server SDK container. We enable the `--bls-composing-models` option which
will also report out stats for the composing models too.

```
sdk:/workspace# perf_analyzer \
    -m translate \
    --input-data data/spanish-news-one.json \
    --measurement-mode=count_windows \
    --measurement-request-count 266 \
    --request-rate-range=0.5:2.0:0.1 \
    --latency-threshold=5000 \
    --max-threads=200 \
    --binary-search \
    -v \
    --bls-composing-models=fasttext-language-identification,seamless-m4t-v2-large
```

After running this, we find that we can handle a maximum request rate of 0.96875
requests/sec.

Request Rate: 0.96875 inference requests per seconds
  * Pass [1] throughput: 0.963679 infer/sec. Avg latency: 1017893 usec (std 16935 usec).
  * Pass [2] throughput: 0.967187 infer/sec. Avg latency: 1016495 usec (std 15745 usec).
  * Pass [3] throughput: 0.970712 infer/sec. Avg latency: 1016811 usec (std 16480 usec).
  * Client:
    * Request count: 798
    * Throughput: 0.967184 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 1017066 usec (standard deviation 16384 usec)
    * p50 latency: 1011253 usec
    * p90 latency: 1041795 usec
    * p95 latency: 1047106 usec
    * p99 latency: 1059062 usec
    * Avg HTTP time: 1017053 usec (send 227 usec + response wait 1016826 usec + receive
      0 usec)
  * Server: 
    * Inference count: 798
    * Execution count: 798
    * Successful request count: 798
    * Avg request latency: 1016388 usec (overhead 48611 usec + queue 213569 usec +
      compute 754208 usec)
  * Composing models: 
    * fasttext-language-identification, version: 2
        * Inference count: 17578
        * Execution count: 17578
        * Successful request count: 17578
        * Avg request latency: 236 usec (overhead 2 usec + queue 20 usec + compute
          input 8 usec + compute infer 198 usec + compute output 6 usec)
    * seamless-m4t-v2-large, version: 3
        * Inference count: 17557
        * Execution count: 1597
        * Successful request count: 17557
        * Avg request latency: 967561 usec (overhead 18 usec + queue 213549 usec +
          compute input 164 usec + compute infer 753653 usec + compute output 175 usec)

