# Tutorial 4: Business Scripting Logic (BLS) - Dynamic Batching
In the previous tutorial, though we enabled dynamic batching in the pbtxt file we
didn't handle the logic of batch processing in `translate.py`. This tutorial will
add that logic in.

As we did before, we will need to alter the logic of the python code to do this. And
like before, this will require a bit more complicated bookkeeping to ensure we send
back the right response to the right request.

Let's make these changes in `model_repository/translate/2/translate.py` since this
is the second version of the model.

## Changes to translate.py
At a high-level, we make the following changes to the `execute(requests)` code.

1. Dynamic Batching Support:
   * Version 1: Assumes a single request per execution.
   * Version 2: Supports multiple requests in a dynamic batch.
   * Why: To improve throughput by processing multiple requests simultaneously.

2. Error Handling:
   * Version 1: Stops processing on the first error encountered.
   * Version 2: Continues processing other requests even if one fails.
   * Why: To increase robustness and ensure all requests are handled.

3. Result Collection:
   * Version 1: Uses a simple list to collect translated chunks.
   * Version 2: Uses a defaultdict to organize results by batch and chunk IDs.
   * Why: To better manage multiple requests and their corresponding chunks.

4. Asynchronous Processing:
   * Version 1: Processes chunks sequentially within each request.
   * Version 2: Submits all chunks from all requests before awaiting results.
   * Why: To maximize parallelism and reduce overall processing time.

5. Logging:
   * Version 1: No logging implemented.
   * Version 2: Includes logging to track the number of requests received.
   * Why: To provide better observability and debugging capabilities.

6. Response Handling:
   * Version 1: Builds responses as it processes each request.
   * Version 2: Prepares a response array and fills it after processing all requests.
   * Why: To accommodate the dynamic batching approach and ensure correct ordering.

7. Language Identification Error Handling:
   * Version 1: Returns immediately on language identification error.
   * Version 2: Marks the request as failed but continues processing others.
   * Why: To increase resilience and process as many requests as possible.

8. Memory Efficiency:
   * Version 1: Creates new lists for each request.
   * Version 2: Preallocates arrays for responses and status flags.
   * Why: To improve memory usage and reduce allocations.

These changes collectively make version 2 more scalable, robust, and efficient in
handling multiple translation requests simultaneously.

To begin, we need to keep track of more things for the entire batch. We add some
additional aggregations to support this:

```
-  responses = []
+  logger = pb_utils.Logger
+  n_requests = len(requests)
+  logger.log_warn(f"`translate` received {n_requests} requests in dynamic batch")
+  responses = [None] * n_requests
+  is_ok = [True] * n_requests
+  inference_response_awaits = []
+  batch_chunk_ids = []
+  results = defaultdict(dict)
```
Before we just needed to gather up the list of `responses`. We still need to do this,
but this time we make the length of `responses` explicitly match the length of
`requests`. Here is also an example of how to get some logging appear to help with
debugging. The `is_ok` will be set to `false` if we get an error somewhere along the
way to try and send skip any subsequent processing and also to ensure we send back an
error message that we can. The `inference_response_awaits` was previously defined
inside the `for` loop for each request, but here we will want to `await` after
submitting all chunks from all requests. Previously, we waited after each document.
The `batch_chunk_ids` will be used in conjunction with `inference_response_awaits` so
we know which `batch_id` and which `chunk_id` is associated with a given inference
response. Lastly, the `results` will gather up the translated text for each chunk of
each document and store it in a dictionary of dictionaries that looks like:

```
{
    <batch_id_0>: {
        <chunk_id_0>: "translated chunk 0 from document 0",
        <chunk_id_1>: "translated chunk 1 from document 0",
        ...,
    },
    <batch_id_1>: {
        <chunk_id_0>: "translated chunk 0 from document 1",
        ...,
    },
    ...,
}
```

As we start to loop through the `requests` we make the following changes to assign
a `batch_id` to each request.

```
- for request in requests:
+ for batch_id, request in enumerate(requests):
```
Like before, we get the input data from the request. If we get an error here, we
handle it a bit differently.

```
try:
    input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
except Exception as exc:
    response = pb_utils.InferenceResponse(
        error=pb_utils.TritonError(
            f"{exc}", pb_utils.TritonError.INVALID_ARG
        )
    )
-   responses.append(response)
+   responses[batch_id] = response
+   is_ok[batch_id] = False
    continue
```

We handle the request parameters just like before and we get to the part where
we will iterate through the chunks of the document.

```
-       for chunk_tt in self.chunk_document(input_text_tt):
+       for chunk_id, chunk_tt in enumerate(self.chunk_document(input_text_tt)):
            if src_lang is None:
                infer_lang_id_request = pb_utils.InferenceRequest(
                    model_name="fasttext-language-identification",
                    requested_output_names=["SRC_LANG"],
                    inputs=[chunk_tt],
                )
                # Perform synchronous blocking inference request
                infer_lang_id_response = infer_lang_id_request.exec()
                if infer_lang_id_response.has_error():
-                   had_error = True
+                   err_msg = (
+                       f"{chunk_id=:} had error: "
+                       + f"{infer_lang_id_response.error().message()}"
+                   )
                    response = pb_utils.InferenceResponse(
-                       error=pb_utils.TritonError(f"{infer_lang_id_response.error().message()"})
+                       error=pb_utils.TritonError(err_msg)
                    )
-                   responses.append(response)
+                   responses[batch_id] = response
+                   is_ok[batch_id] = False
                    break
                src_lang_tt = pb_utils.get_output_tensor_by_name(
                    infer_lang_id_response, "SRC_LANG"
                )
            infer_seamless_request = pb_utils.InferenceRequest(
                model_name="seamless-m4t-v2-large",
                requested_output_names=["TRANSLATED_TEXT"],
                inputs=[chunk_tt, src_lang_tt, tgt_lang_tt],
            )
            # Perform asynchronous inference request
            inference_response_awaits.append(infer_seamless_request.async_exec())
+           batch_chunk_ids.append((batch_id, chunk_id))
-       if had_error:
-           return responses

-       inference_responses = await asyncio.gather(*inference_response_awaits) 
+   # After submitting all the chunks for all the requests, wait for results
+   inference_responses = await asyncio.gather(*inference_response_awaits)
```
Notice that we now call the `await asyncio.gather()` outside of the for-loop through
the requests instead of inside that for-loop. This causes a bigger changes in how
we loop through the `inference_responses`. Previously we simply looped through and
appended the resulting translated text to an array which we then joined together into
the final translated text for that request to send back in a response. Here we loop
through the `infer_seamless_response` and zip it with the `batch_chunk_ids` in order
to build up the `results` dictionary of dictionaries. Once we have all the translated
text for all the chunks for all the requests, we can loop through that to create the
needed responses. We use the `is_ok` array to know if we should proceed with this step
or if there was an error.

```
        for infer_seamless_response, (batch_id, chunk_id) in zip(
            inference_responses, batch_chunk_ids
        ):
            if infer_seamless_response.has_error() and responses[batch_id] is None:
                err_msg = (
                    f"{chunk_id=:} had error: "
                    + f"{infer_seamless_response.error().message()}"
                )
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(err_msg)
                )
                responses[batch_id] = response
                is_ok[batch_id] = False
            else:
                translated_chunk_np = (
                    pb_utils.get_output_tensor_by_name(
                        infer_seamless_response, "TRANSLATED_TEXT"
                    )
                    .as_numpy()
                    .reshape(-1)
                )
                translated_chunk = translated_chunk_np[0].decode("utf-8")
                results[batch_id][chunk_id] = translated_chunk

        for batch_id in sorted(results):
            if is_ok[batch_id]:
                result = results[batch_id]
                translated_chunks = [result[chunk_id] for chunk_id in sorted(result)]
                translated_doc = " ".join(translated_chunks)
                translated_doc_tt = pb_utils.Tensor(
                    "TRANSLATED_TEXT",
                    np.array([translated_doc], dtype=self.translated_text_dtype),
                )
                # Create the response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[translated_doc_tt]
                )
                responses[batch_id] = inference_response

        return responses
```
## Restart Triton Inference Server
With these changes, we can restart the server using

```
$ CONFIG_NAME=tutorial4 docker-compose up
```

At this point, we should be running:

| Model                            | Version | Status |
| :--------------------------------|:-------:|:------:|
| fasttext-language-identification | 2       | READY  |
| seamless-m4t-v2-large            | 3       | READY  |
| translate                        | 2       | READY  |

## Performance Analyzer
With things up and running, let's try out our performance analyzer to see if we can
do better than our 1.1125 inference requests per second from the end of Tutorial 3.

```
sdk:/workspace# perf_analyzer \
    -m translate \
    --input-data data/spanish-news-one.json \
    --measurement-mode=time_windows \
    --measurement-interval=200000 \
    --request-rate-range=1.0:1.36:0.05 \
    --latency-threshold=5000 \
    --max-threads=200 \
    --binary-search \
    -v \
    --bls-composing-models=fasttext-language-identification,seamless-m4t-v2-large
```

I find that we can sustain 1.225 inferences per second. This means that we have gone
from:
  * 0.89375 (translate v1)
  * 1.1125  (translate v1 with 12.5ms delay in seamless-m4t-v2-large dynamic batching)
  * 1.225   (translate v2)

Overall that is a 37% improvement over our initial attempt!

* Request Rate: 1.225 inference requests per seconds
  * Pass [1] throughput: 1.20833 infer/sec. Avg latency: 1975418 usec (std 235794 usec).
  * Pass [2] throughput: 1.22916 infer/sec. Avg latency: 2035321 usec (std 225258 usec).
  * Pass [3] throughput: 1.225 infer/sec. Avg latency: 2071481 usec (std 225203 usec).
  * Client: 
    * Request count: 879
    * Throughput: 1.22083 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 2027653 usec (standard deviation 108447 usec)
    * p50 latency: 1769350 usec
    * p90 latency: 2571546 usec
    * p95 latency: 2615334 usec
    * p99 latency: 2677987 usec
    * Avg HTTP time: 2027641 usec (send 149 usec + response wait 2027492 usec +
      receive 0 usec)
  * Server: 
    * Inference count: 879
    * Execution count: 512
    * Successful request count: 879
    * Avg request latency: 2027226 usec (overhead 1184086 usec + queue 10293 usec +
      compute 832847 usec)

  * Composing models: 
  * fasttext-language-identification, version: 2
      * Inference count: 19382
      * Execution count: 19382
      * Successful request count: 19382
      * Avg request latency: 123 usec (overhead 1 usec + queue 8 usec +
        compute input 4 usec + compute infer 106 usec + compute output 3 usec)

  * seamless-m4t-v2-large, version: 3
      * Inference count: 19338
      * Execution count: 880
      * Successful request count: 19338
      * Avg request latency: 843038 usec (overhead 20 usec + queue 10285 usec +
        compute input 143 usec + compute infer 832417 usec + compute output 172 usec)
