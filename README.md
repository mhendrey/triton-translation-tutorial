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
This will be the simplest of the config.pbtxt files since the model will expect just
a single string be sent to it and it will return a single string. This model is tiny,
fast, and doesn't need a GPU.

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
identification on the different chunks of the document better handle documents that
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

The output field mirrors the input field since the model will send back just a single
answer, that is, it's best guess as to the language of the submitted text.

```
output [
    {
        name: "SRC_LANG"
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
will assume the documents are too long to cause memory issues. Otherwise, the BLS code
will need to handle that logic as well.

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
        There must be a response for each request, but you can return TritonError
        instead of a InferenceResponse if something went wall for the client.

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
the SRC_LANG output variable. In the config.pbtxt, we had stated the data type was
TYPE_STRING. This is just a convenience for when we go to make the `InferenceResponse`.

In the `execute()` method, we loop through the list of requests and handle them one at
a time. This shows the basic pattern that we will use for nearly any model. NOTE: this
may not be the most optimized, but it certain is the simplest.

* Get the input data as Triton Tensors from the request
* If we have an error, create an `InferenceResponse` that has a `TritonError` that
  will be sent back to the client. This is a nice feature to communicate that the
  client sent some bad data.
* Convert the Triton Tensor(s) to more useful objects, mostly through conversion to
  numpy arrays
* Run the input through the model to obtain output data
* Convert the output into a Triton Tensor & make an `InferenceResponse` using that
  Triton Tensor
* Append the `InferenceResponse` to `responses` list
* Return `responses`

One of the unspoken, undocumented features is how the Triton Inference Server knows to
send what response back to which request. As far as I can tell, this is why the
documentation states that the input list of requests and output list of responses need
to the same length. Presumably they need to be the same order too.