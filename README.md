# Tutorial 2: Enabling Dynamic Batching
In the first tutorial, we put together a basic Triton deployment. This handled each
request as it came in. Now we will work to try and optimize the throughput performance.
The first step is to enable Triton Inference Server's dynamic batching capability. This
allows the Triton Inference Server to gather up requests before sending them together
to the backend for processing. This will be advantagous for any parallelizable
processing, particularly of the models running on a GPU. When using dynamic batching,
you can specify how long to wait to accumulate the batch (being mindful not to make this
too long for the client) and the maximum size of the batch (being mindful of VRAM).

Before we begin I'd like to highlight a few Github comments from the developers that I
find helpful in setting up the dynamic batching with the Python backend.

In this [comment](https://github.com/triton-inference-server/server/issues/6740#issuecomment-1890165226) they say:

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
main takeaway (with a minor bug fix)

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
  # so split up the batched_output into individual response outputs for each request batch size
  responses = []
  for i, batch_size in enumerate(all_batch_sizes):
    # Break the large output back into individual outputs for each request batch size
    individual_output = get_batch(batched_output, batch_size, i)
    # Create output tensors mapping each request to each response for their respective batch sizes
    # ...
    responses.append(...)

  assert len(responses) == len(requests)
  return responses
```

In addition, there is one other conceptual change when using dynamic batching. There is
an implicit **additional** dimension to the input/out data that is prepended that
represents the batch size. Let's see what this means with the FastText Language
Identification deployment. 

## FastText Language Identification
Because this model does not use the GPU and does not appear to be multi-threaded, there
is not benefit to changing the deployment to match the pseudocode provide in the
previous section. So we will make minimal changes to the config.pbtxt and the code
itself.

### config.pbtxt
We make just two edits to the previos config.pbtxt file. The first is to change the
max_batch_size from 0 -> 16. This will set the maximum size of the `requests` list sent
to fasttext-language-identification.

The second change is adding the line, `dynamic_batching: {}`, at the end. This enables
dynamic batching with default values. This will be good enough for now.

### fasttext-language-identification.py
We only need to make a slight tweak to the input and output data. This relates to the
change from an expected shape of [1] (1-d array with a single element) to now a
shape of [1, 1] (batch size of 1 and then still our 1-d array with a single element).
We easily take care of this by reshaping the numpy arrays as needed. The rest of the
code can stay the exact same. Here is the diff:

```
-  input_text = input_text_tt.as_numpy()[0].decode("utf-8")
+  input_text = input_text_tt.as_numpy().reshape(-1)[0].decode("utf-8")
 
   src_lang_tt = pb_utils.Tensor(
       "SRC_LANG",
-      np.array([src_lang], dtype=self.src_lang_dtype),
+      np.array([src_lang], dtype=self.src_lang_dtype).reshape(-1, 1),
   )
   src_script_tt = pb_utils.Tensor(
       "SRC_SCRIPT",
-      np.array([src_script], dtype=self.src_script_dtype),
+      np.array([src_script], dtype=self.src_script_dtype).reshape(-1, 1),
   )
```

## SeamlessM4Tv2Large
For this GPU deployment, we will need to make more substantial changes and follow the
pseudocode example above. The point is to create a bigger batch of inputs to be
processed by the model at one time. However, we have a problem with the Transformers
library that we will need to work around.

Take a look at the two of the methods we call, [`processor.__call__()`](https://huggingface.co/docs/transformers/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor.__call__)
and [`model.generate()`](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToText.generate).
Notice that the first only allows you to specify the `src_lang` as a `str` and the
second has `tgt_lang` as a `str`. This means that we cannot have a batch where the
input language varies throughout the batch and also that the output language cannot
vary in the batch either. Ideally, we should be able to provide a `List[str]` where
the list is the size of the batch, just like the `text` and `input_ids` can have a
batch size. This is going to cause an issue for our implementation since we would like
to be agnostic as to the input/output languages that are submitted to the service.
There is likely a way around this if we are willing to go to a lower level of
abstraction, but that defeats the purported simplicity of using the transformers
library. So we leave this as an advance step to the student or submit an issue to
transformers to fix this oversight.

With that giant caveat out of the way, let's plow ahead since our goal is to gain a
better understanding of Triton Inference Server.

### config.pbtxt
