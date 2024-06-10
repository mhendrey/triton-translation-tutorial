# Tutorial 1: Simplest Working Example
This tutorial will specify a minimal config.pbtxt file and Python code to have a
working version of our machine translation service. The service will consist of three
different deployment packages: 

* translate: This is our BLS deployment that serves as the primary interface for
  clients to allow them to send an entire document for translation
  * If the source language is not provided in the client request, send the document
    to fasttext-language-identification to determine it
  * Chunk the document into pieces that can be processed by SeamlessM4T
  * Submit a document's text chunks for translation
  * Combine translated chunks back together to be returned to the client
* fasttext-language-identification: Identify the source language of text sent to
  the deployment
* seamless-m4t-v2-large: Translate the input text from the source language to the
  specified target language

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

### FastText Language Identification
The model will expect just a single string be sent to it and it will return two
strings, the language identified and the script. For Seamless, we only need the
language id, but some other use case may need both. This model is tiny, fast, and
doesn't need a GPU.

Let's review the different parts of the file. We start with the required name and
backend fields. For this whole tutorial, we use the Python backend. We specify
the `max_batch_size` to be zero, which turns off the dynamic batching feature of
Triton Inference Server. We will explore that feature in a few tutorials. We also take
advantage of an optional field, default_model_filename, to use a more descriptive
name that contains the code needed to serve this model.

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
to a numpy array, this will have a dtype of np.object_. 

The last thing we need to specify is the dims section. This is the number of
dimensions of the input data. For this model, we are going to send just a 1-d array
that has just one element in. This will be the entire contents of the document. A
future improvement, that will be left to the student, would be to run language
identification on the different chunks of the document to better handle documents that
contain more than one language in them.

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
environment and that we want to run this model using the CPU.

```
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/fasttext-language-identification.tar.gz"}
}
instance_group [{kind: KIND_CPU}]
```

### SeamlessM4Tv2Large
Since a client may send a large document to be processed, it will likely need to be
chunked into appropriately sized pieces that SeamlessM4Tv2 is meant to handle. To
leverage the parallel processing available in a GPU, these chunks should be submitted
as a 1-d list of variable length size for best performance and then have the
corresponding translated text chunks returned. Remember, we will let the BLS deployment
handle the chunking of the client's document. In addition, for this simple tutorial, we
will assume the documents are not too long to cause memory issues. Otherwise, the BLS
code will need to handle that logic as well.

As a result, the biggest difference will be the dims of the input text and output.
For the FastText model we expected just one element to be sent, hence the `[1]`. But
to handle the variable size here, we set the dims to be `[-1]`. In addition, we need
two other input variables to be provided, SRC_LANG and TGT_LANG. Ideally, these would
match the array length of the INPUT_TEXT, but that's not how the Transformers
SeamlessM4Tv2 works. It takes just a single string for both of these variables. See
[here for src_lang in processor](https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/seamless_m4t_v2#usage) and
[here for tgt_lang](https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model.generate) 

In addition, we specify the instance group to use the GPU instead of a CPU.

### Translate
We have finally arrived at our translate deployment. This will leverage Triton's BLS
to handle the client's input request and then send them along to the needed models for
processing. It's config.pbtxt file looks very similar to the
fasttext-language-identification. In fact even simpler, since this deployment does not
need any special Python libraries.

Like, the FastText model, the translate deployment is expecting a single text string
to be sent by the client and it will return the corresponding translated text back to
the client as the output. This deployment also does not need to be on the GPU, so we
specify that it run on a CPU.

Instead of having the client required to send in the SRC_LANG and TGT_LANG, which are
needed by the SeamlessM4Tv2 deployment, we will let the client send that information
in as optional parameters in their requests. If the client does not pass in these, then
fasttext will be used to determine SRC_LANG and English will be used as the default
TGT_LANG.

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
Based upon the associated config.pbtxt, the SeamlessM4Tv2Large deployment is expecting
to be sent a 1-d array of variable length strings (INPUT_TEXT), a 1-d array of just
one element specifying the source language (SRC_LANG), and a 1-d array of just
one element specifying the target language (TGT_LANG).

The `initialize()` method looks similar to the FastText deployment, but we do take
some care with the loading this onto a GPU. For this model, we need to run both a
processor (handles converting text -> pytorch tokens and the reverse) and the model
itself. Only the model goes onto the GPU.

The `execute()` method also looks very similar. Only the input text is handled a bit
differently given that it is a 1-d array with more than one element.

### Translate
With our two underlying models coded up, we turn our attention to the translate
deployment. This leverages the Business Scripting Logic and allows us to create a more
client-friendly interface to our machine translation server. Again, we begin with the
simplest implementation that we can. Specifically, we will use synchronoous implement
that submits inference requests to the models and waits for their responses back. A
future improvement would be to make these asynchronous requests that could improve
throughput.

The `initialize()` method looks very similar to the others except we don't have any
model that we need to load.

The `execute()` method is where things start looking different from what we have seen
previously. Of course, we still need to iterate over the requests and gather up their
corresponding responses. Like before, we begin processing each request by getting the
intput data out of the client's request. If there is an error here, we capture it and
include it in the response back to the client.

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

Next we see an example of processing option request parameters sent by the client. In
this case, there are two optional parameters, `src_lang` and `tgt_lang`, that a client
may send. If the `src_lang` isn't provide, then we will use the FastText deployment to
determine it. If the `tgt_lang` is provide, then the submitted text will be translated
into that language. Otherwise, it will default to translating into English. Since we
won't be changing the `tgt_lang`, we make the necessary `pb_utils.Tensor` out of it
that will be used to send to the SeamlessM4Tv2Large deployment.

```
request_params = json.loads(request.parameters())
src_lang = request_params.get("src_lang", None)
tgt_lang = request_params.get("tgt_lang", "eng")
tgt_lang_tt = pb_utils.Tensor("TGT_LANG", np.array([tgt_lang], np.object_))
```

The value of the BLS is it's ability to handle branching logic. We next see an example
of that. If the client submitted a `src_lang`, then we don't need to call the FastText
model. If they didn't provide this value, then we get it from FastText. We use
`pb_utils.InferenceReqeust` to make this call to FastText.

```
if src_lang:
    src_lang_tt = pb_utils.Tensor(
        "SRC_LANG", np.array([src_lang], np.object_)
    )
else:
    # Create inference request object
    infer_lang_id_request = pb_utils.InferenceRequest(
        model_name="fasttext-language-identification",
        requested_output_names=["SRC_LANG"],
        inputs=[input_text_tt],
    )

    # Peform synchronous blocking inference request
    infer_lang_id_response = infer_lang_id_request.exec()
    if infer_lang_id_response.has_error():
        response = pb_utils.InferenceResponse(
            error=pb_utils.TritonError(
                f"{infer_lang_id_response.error().message()}"
            )
        )
        responses.append(response)
        continue

    src_lang_tt = pb_utils.get_output_tensor_by_name(
        infer_lang_id_response, "SRC_LANG"
    )
```

In the next code block we break the input text into chunks and then submit them all
at once to the SeamlessM4Tv2Large deployment. We define a class method
`chunk_document()`. In this implementation, we do a simplistic splitting based upon
'.'. This is where future improvements would implement a better chunking strategy.
At the end we use another class method that we define, `combine_translated_chunks()`
that recombines the translated chunks back into a single document to send back to the
client.

```
# Chunk up the input_text_tt into pieces for translation
input_chunks_tt = self.chunk_document(input_text_tt)
# Create inference request object for translation
infer_seamless_request = pb_utils.InferenceRequest(
    model_name="seamless-m4t-v2-large",
    requested_output_names=["TRANSLATED_TEXT"],
    inputs=[input_chunks_tt, src_lang_tt, tgt_lang_tt],
)

# Perform synchronous blocking inference request
infer_seamless_response = infer_seamless_request.exec()
if infer_seamless_response.has_error():
    response = pb_utils.InferenceResponse(
        error=pb_utils.TritonError(
            f"{infer_seamless_response.error().message()}"
        )
    )
    responses.append(response)
    continue

# Get translated chunks
translated_chunks_tt = pb_utils.get_output_tensor_by_name(
    infer_seamless_response, "TRANSLATED_TEXT"
)
# Combine translated chunks
translated_doc_tt = self.combine_translated_chunks(translated_chunks_tt)
```

Finally, we make our `pb_utils.InferenceResponse` to send back to the client.

```
inference_response = pb_utils.InferenceResponse(
    output_tensors=[translated_doc_tt]
)
responses.append(inference_response)
```

With everything now defined, we are ready to launch our service

## Start Triton Inference Service
We launch the service from the parent directory using docker compose

```
$ docker-compose up
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
Take a look a the client_requests.py in examples/ to see how to structure your
requests. Each requests has you specify the input data you are sending which includes:

* name - must match what's in the config.pbtxt file
* shape - The shape of this request. Look at the SeamlessM4Tv2Large example
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
$ docker run --rm -it --net host -v ./data:/workspace/data nvcr.io/nvidia/tritonserver:24.04-py3-sdk
```
This starts up the container and mounts the repo's /data directory which contains the
load testing data we will use. There are a bunch of features in this, but we will
try to keep to this tutorial's ethos, keep it as simple as possible.

From inside the SDK container, let's give the following command a try. We will kick
off the command and then talk about it's pieces while we wait.

```
sdk:/workspace# perf_analyzer \
  -m translate \
  --input-data data/spanish-news-one.json \
  --measurement-mode=count_windows \
  --measurement-request-count=20 \
  --request-rate-range=0.25:4.0:0.1 \
  --latency-threshold=5000 \
  --max-threads=16 \
  --binary-search \
  --bls-composing-models=fasttext-language-identification,seamless-m4t-v2-large
```

As you can see perf_analyzer takes many [arguments](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/docs/cli.html#perf-analyzer-cli). Here are the ones that I'm using

* -m : Specifies the deployment to analye
* --input-data : You can provide your own data, otherwise it will generate random data
* --measurement-mode : Measure for a set number of requests. You could also do time
* --measurement-request-count : Minimum number of requests in the count_window
* --request-rate-range : Gives the min:max:step that will be used in binary search
* --latency-threshold : Maximum time in ms before request rate considered too fast
* --max-threads : Number of threads to use to send the requests
* --binary-search : Perform search between request-rate-range specified
* --bls-composing-models : Gives break down by these models too

This will use a binary search to find what request rate the translate deployment can
handle while keeping the latency below 5 seconds.

For the output we get the following

```
Request Rate: 0.894531 inference requests per seconds
  Client: 
    Request count: 60
    Throughput: 0.869468 infer/sec
    Avg latency: 1082904 usec (standard deviation 22312 usec)
    p50 latency: 1088082 usec
    p90 latency: 1110780 usec
    p95 latency: 1113430 usec
    p99 latency: 1115020 usec
    Avg HTTP time: 1082890 usec (send/recv 120 usec + response wait 1082770 usec)
  Server: 
    Inference count: 60
    Execution count: 60
    Successful request count: 60
    Avg request latency: 1082211 usec (overhead 1913 usec + queue 147 usec + compute 1080151 usec)

  Composing models: 
  fasttext-language-identification, version: 1
      Inference count: 61
      Execution count: 61
      Successful request count: 61
      Avg request latency: 2648 usec (overhead 7 usec + queue 82 usec + compute input 39 usec + compute infer 2478 usec + compute output 41 usec)

  seamless-m4t-v2-large, version: 1
      Inference count: 60
      Execution count: 60
      Successful request count: 60
      Avg request latency: 1077663 usec (overhead 6 usec + queue 65 usec + compute input 36 usec + compute infer 1077528 usec + compute output 26 usec)

Inferences/Second vs. Client Average Batch Latency
Request Rate: 0.25, throughput: 0.247901 infer/sec, latency 1116620 usec
Request Rate: 0.71875, throughput: 0.705769 infer/sec, latency 1091147 usec
Request Rate: 0.835938, throughput: 0.844962 infer/sec, latency 1086519 usec
Request Rate: 0.894531, throughput: 0.869468 infer/sec, latency 1082904 usec
```

So, it looks like our target to beat is 0.869 infer/sec. We also see, that the seamless
model is where all the time is spent. Not surprising given that is where the bulk of
the work is being done.

Since, we aren't sure where we will go for perfomance boosts, let's do a run that
measures just the seamless-m4t-v2-large deployment. We leave off the 
`--bls-composing-models` flag (seg faults otherwise, don't ask me how I know). We
also change the input data since this deployment has different inputs/outputs.

```
perf_analyzer \
  -m seamless-m4t-v2-large \
  --input-data data/spanish-news-seamless-one.json \
  --measurement-mode=count_windows \
  --measurement-request-count=20 \
  --request-rate-range=0.25:4.0:0.1 \
  --latency-threshold=5000 \
  --max-threads=16 \
  --binary-search
```

As we would have hoped, we get the same answers as before. This tells us that we lose
negligible time going through the translate deployment. That's nice, because what
client wants to chunk up their documents and always no the source language before
sending in their request.

```
Inferences/Second vs. Client Average Batch Latency
Request Rate: 0.25, throughput: 0.247903 infer/sec, latency 1097453 usec
Request Rate: 0.71875, throughput: 0.705791 infer/sec, latency 1088433 usec
Request Rate: 0.835938, throughput: 0.844959 infer/sec, latency 1100355 usec
Request Rate: 0.894531, throughput: 0.86946 infer/sec, latency 1090506 usec
```

In the next tutorial, we will explore leveraging Triton Inference Server's dynamic
batching capability. See in branch v2!