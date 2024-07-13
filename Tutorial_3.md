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

## configs.pbtxt
Here is the entire contents of the file. We give our deployment a nice easy name for our
requestors to know. The BLS is supported by the Python backend. Since we know that we
will eventually enable dynamic batching, let's just start off with the config set up for
it, but we won't properly handle the batch in the `.py` file until the next tutorial.
The input is expecting an array with a single string which is the contents of the
document to be translated. The output will the same, but now with the translated text.
Because this deployment will submit requests to the seamless-m4t-v2-large deployment,
it won't need to run on a GPU. Lastly, we are specifically telling Triton Inference
Server to run version `1` of the model.

```
name: "translate"
backend: "python"
max_batch_size: 10
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
version_policy: { specific: { versions: [1]}}
dynamic_batching: { }
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
5. Each chunk is sent to the SeamlessM4Tv2Large model for translation using asynchronous
   requests.
6. The translated chunks are collected and combined into a single document using
    `combine_translated_chunks()`.
7. The final translated document is wrapped in a Triton inference response and added to
   the list of responses to be returned.

This structure allows for efficient processing of large documents by breaking them into
manageable pieces and leveraging asynchronous operations for improved performance.
Though we have enabled dynamic batching within the `config.pbtxt`, we won't implement
any batching logic just yet within the code to keep things simple to start. We will
process a single document at a time. This will likely affect performance just as
before, but at least a single document will be handled efficiently.

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

The BLS can submit inference requests to other Triton deployment endpoints. Here is
an example where we send data to the FastText model for language identification:

```
infer_lang_id_request = pb_utils.InferenceRequest(
    model_name="fasttext-language-identification",
    requested_output_names=["SRC_LANG"],
    inputs=[chunk_tt],
)
# Perform synchronous blocking inference request
infer_lang_id_response = infer_lang_id_request.exec()
```

Again, we leverage existing functionality that is part of the `pb_utils` library by
creating an `InferenceRequest`. We need to specify the `model_name` which is nothing
more than the model endpoint (same as the directory name for the model of interest).
We also specify a list of the outputs that we want with `requested_output_names`.
Notice, that the fasttext-language-identification model returns two outputs,
`SRC_LANG` and `SRC_SCRIPT` [see it's config.pbtxt file], but we only care about the
first.  Of course, we also need to provide the necessary `inputs` which must be
Triton `Tensor` objects. Here we provide the `chunk_tt` which is yielded by the
`chunk_document()`. Be sure that you get the correct shape which for this example
is [1, 1]. Remembering that that first dimension specifies the batch size.

The second line waits for the request to be executed and the result returned is a
`pb_utils.InferenceResponse` object. Because the FastText is so fast and because we
need this information before we can call the SeamlessM4Tv2Large, we do this synchronously
meaning that we wait for the response to come back before we continue.

To get the output of the FastText request, we use `pb_utils.get_output_tensor_by_name()`
which takes as input the `InferenceResponse` and the name of the output we want.

```
src_lang_tt = pb_utils.get_output_tensor_by_name(
    infer_lang_id_response, "SRC_LANG"
)
```

As you are hopefully learning, this returns a `pb_utils.Tensor`. Normally, we would
cast this to a more familiar numpy array, but not this time. Instead, we can use this
directly when we submit our request to the SeamlessM4Tv2Large deployment:

```
infer_seamless_request = pb_utils.InferenceRequest(
    model_name="seamless-m4t-v2-large",
    requested_output_names=["TRANSLATED_TEXT"],
    inputs=[chunk_tt, src_lang_tt, tgt_lang_tt],
)
inference_response_awaits.append(infer_seamless_request.async_exec())
```
This looks just like the FastText `InferenceReqeust`, but of course the variables have
changed to match the correct inputs and outputs available to specified in the pbtxt
file. Only this time, we call `async_exec()` on the `InferenceRequest` in order to
perform asynchronous calls to the SeamlessM4Tv2Large deployment. For now we simply
collect the async tasts into an array and continue the loop.

After we have submitted all the chunks of the document, we `await` for them all to
finish and then loop through the resulting responses. Here we will convert the returned
`pb_utils.Tensor` into a numpy array from which we can then get to the Python string that
we want to gather up.

```
inference_responses = await asyncio.gather(*inference_response_awaits)
for infer_seamless_response in inference_responses:
    if infer_seamless_response.has_error():
        translated_chunks.append(infer_seamless_response.error().message())
    else:
        translated_chunk_np = (
            pb_utils.get_output_tensor_by_name(
                infer_seamless_response, "TRANSLATED_TEXT"
            )
            .as_numpy()
            .reshape(-1)
        )
        translated_chunk = translated_chunk_np[0].decode("utf-8")
        translated_chunks.append(translated_chunk)

```

The resulting translated text, an array of strings, will then be joined together to
create the complete, translated text which is wrapped into a `pb_utils.Tensor`. That
`Tensor` is turned into `pb_utils.InferenceResponse` which will then get returned to
the client that requested the document be translated.

```
translated_doc = " ".join(translated_chunks)
translated_doc_tt = pb_utils.Tensor(
    "TRANSLATED_TEXT",
    np.array([translated_doc], dtype=self.translated_text_dtype),
)
# Create the response
inference_response = pb_utils.InferenceResponse(
    output_tensors=[translated_doc_tt]
)
responses.append(inference_response)
```

With all these changes made, let's restart the service, but this time using the
docker-compose-tutorial3.yaml:

```
$ CONFIG_NAME=tutorial3 docker-compose up
```

At this point, we should be running:

| Model                            | Version | Status |
| :--------------------------------|:-------:|:------:|
| fasttext-language-identification | 2       | READY  |
| seamless-m4t-v2-large            | 3       | READY  |
| translate                        | 1       | READY  |


## Performance Analyzer
Just like before, we can leverage the built in perf_analyzer available in the Triton
Inference Server SDK container. We enable the `--bls-composing-models` option which
will also report out stats for the composing models too.

```
sdk:/workspace# perf_analyzer \
    -m translate \
    --input-data data/spanish-news-one.json \
    --measurement-mode=time_windows \
    --measurement-interval=200000 \
    --request-rate-range=0.85:1.2:0.05 \
    --latency-threshold=5000 \
    --max-threads=200 \
    --binary-search \
    -v \
    --bls-composing-models=fasttext-language-identification,seamless-m4t-v2-large
```

After running this, we find that we can handle a maximum request rate of 0.89375
requests/sec.

Request Rate: 0.89375 inference requests per seconds
  * Pass [1] throughput: 0.887498 infer/sec. Avg latency: 1099948 usec (std 22327 usec). 
  * Pass [2] throughput: 0.895828 infer/sec. Avg latency: 1098067 usec (std 20765 usec). 
  * Pass [3] throughput: 0.891659 infer/sec. Avg latency: 1097961 usec (std 27116 usec). 
  * Client: 
    * Request count: 642
    * Throughput: 0.891662 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 1098656 usec (standard deviation 23537 usec)
    * p50 latency: 1091817 usec
    * p90 latency: 1128772 usec
    * p95 latency: 1135987 usec
    * p99 latency: 1162080 usec
    * Avg HTTP time: 1098641 usec (send 278 usec + response wait 1098363 usec +
      receive 0 usec)
  * Server: 
    * Inference count: 642
    * Execution count: 642
    * Successful request count: 642
    * Avg request latency: 1097902 usec (overhead 53589 usec + queue 228363 usec +
      compute 815950 usec)
  * Composing models: 
  * fasttext-language-identification, version: 2
      * Inference count: 14146
      * Execution count: 14146
      * Successful request count: 14146
      * Avg request latency: 299 usec (overhead 3 usec + queue 24 usec +
        compute input 11 usec + compute infer 251 usec + compute output 9 usec)
  * seamless-m4t-v2-large, version: 3
      * Inference count: 14125
      * Execution count: 1285
      * Successful request count: 14125
      * Avg request latency: 1044034 usec (overhead 17 usec + queue 228339 usec +
        compute input 153 usec + compute infer 815371 usec + compute output 153 usec)

Notice that the seamless-m4t-v2-large is doing 14,125 inferences across 1,285
executions. This means that the average batch size is 11 document chunks.  However, we
know that the average document has 22 chunks (14,125 / 642).  This means that each
document is processing two batches. Let's see what happens if we alter the
tutorial3.pbtxt file to have the dynamic batching wait for 12.5 microseconds to
accumulate the batch.

Edit the `model_repository/seamless-m4t-v2-large/configs/tutorial3.pbtxt` by adding the
`max_queue_delay_microseconds` option to `dynamic_batching` so that it now looks like:

```
dynamic_batching: {
  max_queue_delay_microseconds: 12500
}
```
Restart the Triton Inference Server so that the new config changes are loaded and then
rerun the Performance Analyzer.

```
$ CONFIG_NAME=tutorial3 docker-compose up
```

With this small delay we find that we can increase the inference requests per second
from 0.89375 -> 1.1125 which is a 24% improvement. The average batch size also
increased from 11 (14,125 / 1,285) to 21.1 (17,589 / 832)

* Request Rate: 1.1125 inference requests per seconds
  * Pass [1] throughput: 1.10416 infer/sec. Avg latency: 1098300 usec (std 138938 usec). 
  * Pass [2] throughput: 1.11249 infer/sec. Avg latency: 1088880 usec (std 133179 usec). 
  * Pass [3] throughput: 1.1125 infer/sec. Avg latency: 1098245 usec (std 143023 usec). 
  * Client: 
    * Request count: 799
    * Throughput: 1.10972 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 1095134 usec (standard deviation 138335 usec)
    * p50 latency: 1093920 usec
    * p90 latency: 1286280 usec
    * p95 latency: 1314654 usec
    * p99 latency: 1346206 usec
    * Avg HTTP time: 1095121 usec (send 183 usec + response wait 1094938 usec +
      receive 0 usec)
  * Server: 
    * Inference count: 799
    * Execution count: 799
    * Successful request count: 799
    * Avg request latency: 1094568 usec (overhead 217749 usec + queue 19781 usec +
      compute 857038 usec)
  * Composing models: 
  * fasttext-language-identification, version: 2
      * Inference count: 17600
      * Execution count: 17600
      * Successful request count: 17600
      * Avg request latency: 162 usec (overhead 2 usec + queue 11 usec +
        compute input 5 usec + compute infer 139 usec + compute output 4 usec)
  * seamless-m4t-v2-large, version: 3
      * Inference count: 17589
      * Execution count: 832
      * Successful request count: 17589
      * Avg request latency: 876677 usec (overhead 18 usec + queue 19770 usec +
        compute input 153 usec + compute infer 856576 usec + compute output 158 usec)

## Next Steps
In this tutorial, though we have enabled dynamic batching for the translate Triton
deployment, the Python code is only looping through each request and translating them
one request at a time. In the next tutorial, we will make a new version of the
deployment that will asynchronously submit each chunk for **all** documents in the
batch to see if we can get a better performance than 1.1125 documents / second.