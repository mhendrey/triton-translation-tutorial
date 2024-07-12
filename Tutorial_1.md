# Tutorial 1: The Basics
This tutorial will specify a minimal config.pbtxt file and associated Python code
to stand up a simple machine translation service. For this basic service, we will
stand up two different deployment packages:

* fasttext-language-identification: Identify the language of text sent to it
* seamless-m4t-v2-large: Translate the input text from the source language to a
  specified target language

The basic structure of a model repository is:

```
<model-repository-path>/
  <model-1-name>/
    config.pbtxt
    1/             # Version 1
      model.py
    2/
      model.py     # Version 2
  <model-2-name>/
    config.pbtxt
    1/
      model.py
    ...
  ...
```

We will review the contents of the config.pbtxt files and the Python code needed.
After that we will start the Triton Inference Server and show how to send some HTTP
requests to the running inference endpoints. Lastly, we will use the Perf Analyzer
command line tool to get some baseline throughput metrics.

## config.pbtxt
The config.pbtxt file is a ModelConfig protobuf that provides required and optional
information about a model in a model repository. The minimal required fields in the
config.pbtxt file are the name, platform, max_batch_size, input, and output, which provide essential information about the model.

See [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) for more details.

It is possible to have some of this information
[generated automatically](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#auto-generated-model-configuration),
but that is not covered in this tutorial.

The input and output portions each must specify a name, datatype, and shape. The name
of the input or output ***must*** match the name expected by the model in your .py
file.

In this tutorial, we will also leverage the optional `configs/` in the model
repository for each model. This will allow us to specify alternative configurations
that can be loaded when we launch the triton inference server container. Normally,
this seems to be done for different architectures. For example, running on 8xH100s
might have on configuration file and running on a single A10 might have a different
configuration file. But for us, we will use this to be able cleanly keep the tutorials
separated from each other as we increase the complexity of our deployment.

However to start, we will use the default `config.pbtxt` file to specify the model
configurations for this first tutorial.

### FastText Language Identification
The model will expect just a single string be sent to it and it will return two
strings, the language identified and the script. For Seamless, we only need the
language id, but some other use case may need both. This model is tiny, fast, and
doesn't need a GPU.

Let's review the different parts of the file. We start with the required name and
backend fields. For this whole tutorial, we use the Python backend. We specify
the `max_batch_size` to be zero, which turns off the dynamic batching feature of
Triton Inference Server. We will explore that feature in the next tutorial. We also
take advantage of an optional field, default_model_filename, to use a more descriptive
name than just `model.py` that contains the code needed to serve this model.

```
name: "fasttext-language-identification
backend: "python"
max_batch_size: 0
default_model_filename: "fasttext-language-identification.py"
```

For the input/output sections, there seems to be a convention of using ALL CAPS to
name the inputs and outputs, so we use `INPUT_TEXT` to be the name for our input.

For input & output data, Triton Inference Server has its own Triton Tensors. You can
see how their data types get mapped to the different backends [here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#datatypes).
Fortunately, the Triton Tensor class has `as_numpy()` method to map to a more familiar
numpy array. You will see this used in the .py files to come. Since we will be sending
in a string, we specify `TYPE_STRING`, but be **forewarned**. Notice that the chart
states that this will be treated as `bytes`. This means we will need to treat the
incoming data as `bytes` in our code. Also, notice that when we map the Triton Tensor
to a numpy array, this will have a dtype of `np.object_`. 

The last thing we need to specify is the dims section. This is the number of
dimensions of the input data. For this model, we are going to send just a 1-d array
that has just one element in. This will be the entire contents of the string whose
language is to be identified.

```
input [
    {
        name: "INPUT_TEXT"
        data_type: TYPE_STRING
        dims: [1]
    }
]
```

The output fields mirrors the input field since the model will send back just a single
answer, that is, it's best guess as to the language id and script of the submitted
text.

```
output [
    {
        name: "SRC_LANG"
        data_type: TYPE_STRING
        dims: [1]
    },
    {
        name: "SRC_SCRIPT"
        data_type: TYPE_STRING
        dims: [1]
    }
]
```

Lastly we add in some optional fields in the config.pbtxt that specify the conda
environment and that we want to run this model using the CPU. In addition, we specify
a specific version of the model to be run, the first version.

```
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/fasttext-language-identification.tar.gz"}
}
instance_group [{kind: KIND_CPU}]
version_policy: { specific: { versions: [1]}}
```

### SeamlessM4Tv2Large
This is the deployed model that performs the translation. Like many translation models,
SeamlessM4Tv2 needs to have the input text chunked into reasonably sized pieces.
Typically, this is done at the sentence level, but in reality you could possible do a
few short sentences together. It seems that the total sequence length that the model
can [handle is 1024 tokens](https://github.com/facebookresearch/seamless_communication/blob/81aee56003022af2c0bc05203bfbd598de73d899/src/seamless_communication/inference/generator.py#L72).
If you provide too much text, then it just uses the last 1024 tokens.

We begin like we did before with specifying:

```
name: "seamless-m4t-v2-large"
backend: "python"
max_batch_size: 0
default_model_filename: "seamless-m4t-v2-large.py"
```

The inputs and outputs look like what we had before as well, with an INPUT_TEXT that is
a TYPE_STRING that has a 1-d array with just a single-element.  In addition, to the
INPUT_TEXT, we need to specify the starting language, SRC_LANG, and what the language
we want to translate to, TGT_LANG. These have the same 1-d arrays of TYPE_STRING with
just a single element each.

```
input [
  {
    name: "INPUT_TEXT"
    data_type: TYPE_STRING
    dims: [1]
  },
  {
    name: "SRC_LANG",
    data_type: TYPE_STRING
    dims: [1]
  },
  {
    name: "TGT_LANG",
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
```

Lastly, we specify the conda pack to use and for this deployment we want it to run on
GPU. To explicitly require it run on a GPU we would set `kind: KIND_GPU`, but if we
use `kind: KIND_AUTO` then it will run on a GPU if available, otherwise fall back
to running on a CPU. This is nice if you are developing on a non-gpu, but want to
deploy on a GPU. Just remember to also do a similar check in the actual code itself
when we load the model from disk.

```
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/seamless-m4t-v2-large.tar.gz"}
}
instance_group [{ kind: KIND_AUTO }]
version_policy: { specific: { versions: [1]}}
```

## Python Backend
Since we are leveraging the [Python backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html),
we need to create a Python file that has the following minimum structure. See
[here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#usage) for more details.

```
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        # Use this to load your model into memory
        self.model = load_model()
    
    def execute(self, requests):
        """
        Must be implemented. Process incoming requests and return responses.
        There must be a response for each request, but you can return an 
        InferenceResponse that is a TritonError back to the client if something went
        wrong. This way the client gets the error message.

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]

        Returns
        -------
        List[pb_utils.InferenceResponses]
        """
        responses = []
        for request in requests:
            # Get input data out of requests
            output = self.model(input)
            # Make InferenceResponse
            response = ...
            responses.append(response)

        return responses        
```

With that, we need to discuss the single biggest drawback that I can see. The library,
`triton_python_backend_utils`, that we import only exists within the Triton Inference
Server containers. There is no library for you to install and worse, there is seemingly
no documentation that you can reference. This is a seriously drawback that makes using
the Python backend challenging. The best that I've found so far is to go looking at the
[source code](src/resources/triton_python_backend_utils.py) which is written in C++.
The only saving grace is that we don't need to use too many features from this
undocumented, hidden Python package.

### FastText Language Identification
Given this is just the first tutorial, we will be working in version 1 of this model.
This is found in
`model_repository/fasttext-language-identification/1/fasttext-language-identification.py`.
As expected, the `initialize()` loads the FastText model into memory. In addition, we
use the `pb_utils.get_output_config_by_name()` and `pb_utils.triton_string_to_numpy()`
in conjunction to save off the corresponding numpy dtype (np.object_ in this case) for
the SRC_LANG & SRC_SCRIPT outputs. In the config.pbtxt, we had stated the data type was
TYPE_STRING. This is just a convenience for when we go to make the `InferenceResponse`.

In the `execute()` method, we loop through the list of requests and handle them one at
a time. This shows the basic pattern that we will use for nearly any model. NOTE: this
may not be the most optimized, but it certain is the simplest. For each request, we
start by getting the data that was sent. We put this into a try/except which enables
sending an input data error message back through the response.

```
try:
    input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
except Exception as exc:
    response = pb_utils.InferenceResponse(
        error=pb_utils.TritonError(
            f"{exc}", pb_utils.TritonError.INVALID_ARG
        )
    )
    responses.append(response)
    continue

```

Notice that we retrieve the needed data by the name. This is the name given in the
config.pbtxt file. The input_text_tt is a `pb_utils.Tensor` so we next convert that
to a more amenable type. The `pb_utils.Tensor` has the `as_numpy()` method to easily
convert to more standard numpy arrays. Since that the config.pbtxt specifies that
INPUT_TEXT has `dims: [1]`, we have just a 1-d numpy array that has just a single
element. Given that config.pbtxt for INPUT_TEXT set `data_type: TYPE_STRING` you
would think that the resulting numpy array had strings. It does **not**. The array
contains `bytes` and must thus be decoded.

```
input_text = input_text_tt.as_numpy()[0].decode("utf-8")
# Replace newlines with ' '. FastText breaks on \n
input_text_cleaned = self.REMOVE_NEWLINE.sub(" ", input_text)
```

With the data in a format/type that the FastText model can handle, we call the model
to predict the language and script of the input text.

```
try:
    output_labels, _ = self.model.predict(input_text_cleaned, k=1)
except Exception as exc:
    response = pb_utils.InferenceResponse(
        error=pb_utils.TritonError(f"{exc}")
    )
    responses.append(response)
    continue
src_lang, src_script = output_labels[0].replace("__label__", "").split("_")
```
Once again we wrap this in try/except. If we encounter any error, we pass that error
back to the client via the `InferenceResponse`. We take the resulting output and get
the top source language and script predicted.

Finally, we need wrap the model's outputs into `pb_utils.Tensor` and put them into
the `pb_utils.InferenceResponse` to be sent back to the client.

```
src_lang_tt = pb_utils.Tensor(
    "SRC_LANG",
    np.array([src_lang], dtype=self.src_lang_dtype),
)
src_script_tt = pb_utils.Tensor(
    "SRC_SCRIPT",
    np.array([src_script], dtype=self.src_script_dtype),
)
response = pb_utils.InferenceResponse(
    output_tensors=[src_lang_tt, src_script_tt],
)
```

One of the unspoken, undocumented features is how the Triton Inference Server knows to
send what response back to which request. As far as I can tell, this is why the
documentation states that the input list of requests and output list of responses need
to the same length. Presumably they need to be the same order too.

### SeamlessM4Tv2Large
Based upon the associated config.pbtxt, the SeamlessM4Tv2Large deployment is also
expecting to be sent a 1-d array of with one element of the text to be translated
(INPUT_TEXT), a 1-d array of just one element specifying the source language
(SRC_LANG), and a 1-d array of just one element specifying the target language
(TGT_LANG).

The `initialize()` method looks similar to the FastText deployment, but we do take
some care with the loading this onto a GPU. For this model, we need to run both a
processor (handles converting text -> pytorch tokens and the reverse) and the model
itself. Only the model goes onto the GPU. Notice that we are using a modified version
of the SeamlessM4T classes that can handle having batches of text where each element
in the batch can be a different source language and be translated to different target
languages. The Transformer versions of these classes do **not** support that for some
inexplicable reason, though these are pretty straightfoward to fix. For the curious,
the code is in the seamless_fix.py file along side the seamless-m4t-v2-large.py.

```
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    self.device = torch.device("cuda")
    torch_dtype = torch.float16
else:
    self.device = torch.device("cpu")
    torch_dtype = torch.float32  # CPUs can't handle float16
self.model = SeamlessM4Tv2ForTextToTextMulti.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    device_map="auto",
    torch_dtype=torch_dtype,
    cache_dir="/hub",
    local_files_only=True,
)
self.processor = SeamlessM4TProcessorMulti.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    cache_dir="/hub",
    local_files_only=True,
)
```

The `execute()` method looks very similar as well. It receives a list of `requests`
with each request containing the text to be translated, the source langauge of that
text, and the target language for the translation. These are retrieved as
`pb_utils.Tensor` and then converted to lists of Python strings. If we get an error
retrieving the data from the request, we pass that error back in the
`pb_utils.InferenceRequest`.

For the translation part, we do that in three stages. If there is an error at any one
of the stages, we pass that back to the requestor.

1. Use `processor()` to tokenize the input text
2. Use `model.generate()` to get output tokens of the translated text
3. Use `processor.batch_decode()` to decode the output tokens into the translated text

Finally, the translated text is converted back into a `pb_utils.Tensor` and then used
to create the `pb_utils.InferenceResponse` object that is appended to the list of
`responses` which gets returned at the end of `execute()`. Each response corresponds to
a request and contains either the translated text or an error message.

## Start Triton Inference Service
We launch the service from the parent directory using docker compose. We will leverage
the `--model-config-name` option when launching
[tritonserver](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#custom-model-configuration)
We will do this by using an environment variable, `CONFIG_NAME`, when calling docker
compose. The observant student will notice that there is no file named
`configs/tutorial1.pbtxt` which is the file that tritonserver will be looking for. But
if a pattern is not found, then tritonserver loads the default `config.pbtxt` file,
which is what we are doing here.

```
$ CONFIG_NAME=tutorial1 docker-compose up
```

This mounts two volumes into the Triton Inference Server container. The first is the
model repository which contains all the work we just described. The second is the
location of our local huggingface hub cache directory which we use to load the models.

If we are successful you should see:
```
 Started GRPCInferenceService at 0.0.0.0:8001
 Started HTTPService at 0.0.0.0:8000
 Started Metrics Service at 0.0.0.0:8002
```

## Example Request
Take a look a `examples/client_requests.py` to see how to structure your requests. Each
request has you specify the input data you are sending which includes:

* name - must match what's in the config.pbtxt file
* shape - The shape of this request.
* datatype - Since we are using string, this is "BYTES" *shrug*
* data - Array of the input data

```
"name": "SRC_LANG",
"shape": [1],
"datatype": "BYTES",
"data": ["spa"],
```

## Triton Performance Analyzer
The [Triton Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/README.html#triton-performance-analyzer)
is a CLI tool which can help you measure and optimize the inference performance of
your Triton Inference Server. It is available in the SDK container we all ready pulled.
The documentation says you can try `pip install tritonclient` which should have it, but
that you will likely be missing necessary system libraries. They weren't wrong. I tried
on my Ubuntu machine and no luck.

```
$ docker run --rm -it --net host -v ./data:/workspace/data nvcr.io/nvidia/tritonserver:24.05-py3-sdk
```
This starts up the container and mounts the repo's /data directory which contains the
load testing data we will use. There are a bunch of features in this, but we will
try to keep to this tutorial's ethos, keep it as simple as possible.

From inside the SDK container, let's give the following command a try. We will kick
off the command and then talk about it's pieces while we wait.

```
sdk:/workspace# perf_analyzer \
  -m seamless-m4t-v2-large \
  --input-data data/spanish-news-seamless-one.json \
  --measurement-mode=count_windows \
  --measurement-request-count=266 \
  --request-rate-range=1.0:4.0:0.1 \
  --latency-threshold=5000 \
  --max-threads=16 \
  --binary-search \
  -v \
  --stability-percentage=25
```

As you can see perf_analyzer takes many [arguments](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/docs/cli.html#perf-analyzer-cli). Here are the ones that I'm using

* -m : Specifies the deployment to analye
* --input-data : You can provide your own data, otherwise it will generate random data
* --measurement-mode : Measure for a set number of requests. You could also do time
* --measurement-request-count : Minimum number of requests in the count_window
* --request-rate-range : Gives the min:max:step that will be used in binary search
* --latency-threshold : Maximum time in ms a request should take
* --max-threads : Number of threads to use to send the requests
* --binary-search : Perform search between request-rate-range specified
* -v : verbose mode gives more info
* --stability-percentage : Max. % diff between high & low latency for last 3 trials

This will use a binary search to find what request rate the translate deployment can
handle while keeping the latency below 5 seconds.

For the output we get the following

```
Request Rate: 2.5 inference requests per seconds
  Pass [1] throughput: 2.48574 infer/sec. Avg latency: 670880 usec (std 104532 usec). 
  Pass [2] throughput: 2.50443 infer/sec. Avg latency: 663555 usec (std 162182 usec). 
  Pass [3] throughput: 2.47199 infer/sec. Avg latency: 570518 usec (std 256319 usec). 
  Client: 
    Request count: 801
    Throughput: 2.48734 infer/sec
    Avg client overhead: 0.00%
    Avg latency: 634975 usec (standard deviation 115212 usec)
    p50 latency: 424363 usec
    p90 latency: 1247050 usec
    p95 latency: 1337742 usec
    p99 latency: 1519219 usec
    Avg HTTP time: 634960 usec (send 101 usec + response wait 634859 usec + receive 0 usec)
  Server: 
    Inference count: 801
    Execution count: 801
    Successful request count: 801
    Avg request latency: 634382 usec (overhead 6 usec + queue 280831 usec + compute input 35 usec + compute infer 353468 usec + compute output 41 usec)

Inferences/Second vs. Client Average Batch Latency
Request Rate: 1.00, throughput: 1.00 infer/sec, latency 345138 usec
Request Rate: 2.50, throughput: 2.49 infer/sec, latency 634975 usec
```

So, it looks like our target to beat is 2.50 infer/sec for the seamless-m4t-v2-large
deployment.

For comparison, the fasttext-language-identification deployment can support about
2,100 infer/sec.

In the next tutorial, we will explore leveraging Triton Inference Server's dynamic
batching capability which promises to improve that throughput.