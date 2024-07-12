# Tutorial 2: Dynamic Batching
One of the advantages of using Triton Inference Server is it's ability to do
[dynamic batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher)
of incoming requests with minimal effort on the part of the coder.

In this tutorial, to keep things separated from Tutorial 1 in the model repository,
we will specify the config file in `model_repository/{model}/configs/dynamic.pbtxt`
and the code can be found in the version 2 `model_repository/{model}/2` directory.

## Configuration Changes
To enable default dynamic batching, we make just two minor changes to the config file.
For this tutorial, to keep things separated, we will make a new configuration file that
can be found in the `model_repository/{model}/configs/tutorial2.pbtxt`. We make the
following changes compared to the `model_repository/{model}/config.pbtxt`:

```
- max_batch_size: 0
+ max_batch_size: 50                              # Increase batch size for dynamic batch
- version_policy: { specific: { versions: [1]}}
+ version_policy: { specific: { versions: [2]}}   # Change from running version 1 -> 2
+ dynamic_batching: {}                            # Enable default dynamic batching
```

That's it! This enables default dynamic batching with a max request batch size of 50.
You can specify some options inside the `dynamic_batching`. [For example](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#delayed-batching), you can
specify `max_queue_delay_microseconds` which states how long to hold a request in the
queue to allow time for a batch to form.
The last line just says that we should only load the most recent version of the model
when we launch Triton Server.

We make this change for both the fasttext-language-identification and
seamless-m4t-v2-large deployments.

## Coding Changes
If we simply used the previous version of the model, you will find that it throws an
error. This is because the shape of the Triton Tensor inputs/outputs has changed
subtly. With dynamic batching enabled, we now have another dimension to our Tensors
that corresponds to the batch size. This is prepended to the previous dimensions.
This is added automatically which is why we could leave the configuration file's
`shape: [1]` unaltered.  In reality now, our incoming Tensors have shape of [1, 1]
where the first `1` represents the batch size.

This causes us to make the following minor changes to our previous code (stored in
version `2` for each model). For each of the input Tensors and output Tensors, we need
to reshape them. In pseudocode it will be something like:

```
- input_tt.as_numpy()
+ input_tt.as_numpy().reshape(-1)

  output_tt = pb_utils.Tensor(
    "OUTPUT",
-   np.array(output, dtype=self.output_dtype),
+   np.array(output, dtype=self.output_dtype).reshape(-1, 1),
  )
```

## Relaunching Triton Inference Server
To launch the server, we make a minor change to the `docker-compose-tutorial1.yaml`
file by passing in the `--model-config-name=tutorial2` option when starting
tritonserver and saving it as `docker-compose-tutorial2.yaml`. This will force Triton
Server to look in the `configs` folder for our `tutorial2.pbtxt` files. If a model
doesn't have that file, it falls back to the default `config.pbtxt`. See
[here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#custom-model-configuration)
for more info. Make that change and then start up the service with:

```
$ CONFIG_NAME=tutorial2 docker-compose up
```

## Performance Analyzer
Well that certainly seems really simple. Let's see how much that helps our throughput
performance by rerunning the Performance Analyzer from inside the Triton Inference
Server SDK container:

```
sdk:/workspace# perf_analyzer \
  -m seamless-m4t-v2-large \
  --input-data data/spanish-news-seamless-one.json \
  --measurement-mode=count_windows \
  --measurement-request-count=266 \
  --request-rate-range=1.0:30.0:0.5 \
  --latency-threshold=5000 \
  --max-threads=200 \
  --binary-search \
  -v \
  --stability-percentage=25
```

Well that was no good. We still get just the 2.5 infer/sec we had before. Why?

## Dynamic Batching with Python Backend
Turns out, though we have dynamic batching enabled, that we aren't actually batching
up the requests before sending them to the GPU for processing. That's because we are
still looping through each of the requests and calling the `model.generate()` on each
request individually. No wonder we don't see any improvement in throughput.

This [Github comment](https://github.com/triton-inference-server/server/issues/6740#issuecomment-1890165226)
from the developers gives a nice explanation on how to set up dynamic batching for the
Python backend.

> To use dynamic batching with the Python backend in Triton, you need to understand
> that the Python backend is unique compared to some of the other backends (like
> TensorFlow, PyTorch, etc.). It doesn't currently implement the actual "batching"
> logic for you. It supports dynamic batching in the sense that Triton can gather the
> requests in the server/core and then send those requests together in a single API
> call to the model.

Which lays out the basic construct of what we need to do. In the first tutorial, though
we were looping through `requests` these were always of size one without dynamic
batching enabled. Furthermore, calling our expense GPU model on each loop through isn't
efficient either.

This [comment](https://github.com/triton-inference-server/server/issues/5926#issuecomment-1585161393)
does a great job of laying out the overall approach that we want to take. Here's the
main takeaway (with a minor bug fix I noticed)

```
def initialize(self, args):
  # ...
  self.model = MyModel()

def execute(self, requests):
  large_input = []
  all_inputs = []
  all_batch_sizes = []
  for request in requests:
    np_input = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
    batch_size = np_input.shape[0]
    all_inputs.append(np_input)
    all_batch_sizes.append(batch_size) # Commenter forgot this line
  
  # Gather the request inputs into a single batch
  batched_input = np.stack(all_inputs, axis=0)

  # Compute large batch all together into single large output batch
  batched_output = self.model.infer(batched_input)
  
  # Python model execute() must return a response for each request received,
  # so split up the batched_output into individual response outputs for each request
  # batch size
  responses = []
  for i, batch_size in enumerate(all_batch_sizes):
    # Break the large output back into individual outputs for each request batch size
    individual_output = get_batch(batched_output, batch_size, i)
    # Create output tensors mapping each request to each response for their respective
    # batch sizes
    # ...
    responses.append(...)

  assert len(responses) == len(requests)
  return responses
```

## Coding Changes
We will leave the FastText model alone for two reasons. First, it is significantly
faster than the SeamlessM4Tv2. Second, the model itself runs on a cpu and doesn't seem
to utilize multiple processors. Instead, we will focus our attention to making a new
version, `3`, for the SeamlessM4Tv2 model only. We will follow the basic outline given
above.

At a high-level, we will loop through the requests and gather up the input data. When
we are done, we can create the necessary batches (`input_texts`, `src_langs`,
`tgt_langs`) to be used.

With the batched data ready, we run through the tokenization process using the
`processor` and then feed that output into the `model.generate()`. This is the part
that actually runs on the GPU. Finally, we use `processor.batch_decode()` to take
the model output and decode that into text. If we have any errors in these sections, we
need to fail the entire batch. **NOTE** Care should be taken not to send any error
message back to an individual requestor that might contain data from another user.

Finally, we iterate through all of our successfully run translations and generate the
Triton Tensor that is to be returned to each requestor.

The main changes from version `2` to version `3` are focused on improving efficiency by
implementing batch processing. Here are the key differences:

1. Initialization:
   - `execute_v3` initializes a responses list with None values for all requests upfront.
   - It also creates separate lists for valid requests and batch inputs.

2. Input Processing:
   - `execute_v2` processes each request individually in a loop.
   - `execute_v3` collects all valid inputs into batches (batch_input_text, batch_src_lang, batch_tgt_lang).

3. Error Handling:
   - `execute_v2` immediately creates and appends error responses for invalid requests.
   - `execute_v3` skips invalid requests but keeps track of their indices, allowing for batch processing of valid requests.

4. Batch Processing:
   - `execute_v3` combines all valid inputs into single lists (input_texts, src_langs, tgt_langs) using itertools.chain.

5. Model Operations:
   - `execute_v2` performs tokenization, generation, and decoding for each request separately.
   - `execute_v3` performs these operations once on the entire batch of valid requests.

6. Error Propagation:
   - In `execute_v3`, if an error occurs during batch processing, it's propagated to all valid requests in the batch.

7. Response Creation:
   - `execute_v2` creates and appends responses sequentially.
   - `execute_v3` assigns responses to the pre-initialized list using the tracked indices of valid requests.

8. Return Value:
   - Both methods return a list of responses, but `execute_v3` ensures that the order of responses matches the order of incoming requests, including None or error responses for invalid requests.

These changes allow for more efficient processing of multiple requests by leveraging batch operations, which can significantly improve performance, especially for larger numbers of requests.

With these changes made and saved as version `3`, we need to edit the
`seamless-m4t-v2-large/configs/tutorial2.pbtxt` so that when we restart we use this
new version of the model.

```
- version_policy: { specific: { versions: [2]}}
+ version_policy: { specific: { versions: [3]}}
```

Restart the service with the same command as before

```
$ CONFIG_NAME=tutorial2 docker-compose up
```

## Performance Analyzer - Part 2
Let's retry our performance analyzer run. For this run we increase the starting point
for the requests/sec to 30 and set the upper end to 58. Since we are starting at 30, we
also increase the number of requests used in each trial run to 6004. This correspondes
to the low end requests/sec taking about 3.3 minutes per trial. We also increase the
max-threads in order to be able to push enough requests/sec that we are asking for.
Here's the full command line argument:

```
sdk:/workspace# perf_analyzer \
  -m seamless-m4t-v2-large \
  --input-data data/spanish-news-seamless-one.json \
  --measurement-mode=count_windows \
  --measurement-request-count=6004 \
  --request-rate-range=30:58:5 \
  --latency-threshold=5000 \
  --percentile=50 \
  --max-threads=300 \
  --binary-search \
  --stability-percentage=20 \
  -v
```

SUCCESS! We find that we can support 44 requests/sec keeping the median latency for a
single request under 5 seconds.  Here's is the detailed information for that test.

Request Rate: 44 inference requests per seconds
  * Pass [1] throughput: 43.5389 infer/sec. p50 latency: 1511563 usec
  * Pass [2] throughput: 43.8404 infer/sec. p50 latency: 1437355 usec
  * Pass [3] throughput: 44.2131 infer/sec. p50 latency: 1439022 usec
  * Client: 
    * Request count: 18074
    * Throughput: 43.8625 infer/sec
    * Avg client overhead: 0.00%
    * p50 latency: 1461702 usec
    * p90 latency: 3754267 usec
    * p95 latency: 4346325 usec
    * p99 latency: 5245158 usec
    * Avg HTTP time: 1773992 usec (send 66 usec + response wait 1773926 usec + receive 0 usec)
  * Server: 
    * Inference count: 18074
    * Execution count: 570
    * Successful request count: 18074
    * Avg request latency: 1773496 usec (overhead 27 usec + queue 905925 usec + compute input 257 usec + compute infer 867042 usec + compute output 243 usec)

In fact, we see that the average latency for a single request is just 1.77 seconds
where it spends 0.9 seconds in the queue and 0.867 seconds running on the GPU. The
average batch size was 31.7 (18074 / 570).