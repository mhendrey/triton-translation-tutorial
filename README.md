# Triton Translation Tutorial
In this tutorial, we will build a machine translation service using [NVIDIA's Triton
Inference Server](https://developer.nvidia.com/triton-inference-server) to run two
models, both from Meta. The first is the
[FastText Language Identification](https://huggingface.co/facebook/fasttext-language-identification)
model to automatically determine what language is provide by the user. The second is
the [SeamlessM4Tv2Large](https://huggingface.co/facebook/seamless-m4t-v2-large) to
perform the translation. We will start with a basic deployment to introduce core
concepts and iteratively refine our approach to optimize the translation services
throughput. Each section of the tutorial will be a separate branch.

Key concepts that will be covered:
  * Use the Python backend to deploy a model
  * Use [Business Logic Scripting](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#business-logic-scripting) (BLS) to act as a service level deployment to make a client-friendly interface
  * Enable Dynamic Batching
  * Use the provided (in the SDK container) perf_analyzer to measure performance of the deployment

## Overall Goal
We will build a service that let's a client send a document that needs to be translated
to the BLS endpoint. Since the document will likely be too long for SeamlessM4T to
process, the BLS deployment will handle document chunking and then submit a batch of
text chunks to the seamless model for translation. The batching will help with throughput.
If the client does not specify the source language of the document, the BLS deployment
will first send the document to the fasttext language identification model.

After the end of each tutorial session (i.e., branch), we will use some
[Spanish news articles](https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification)
to measure the servers throughput performance. This data is stored in the data
directory and follows the formatting need for the
[perf_analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/docs/README.html)
cli provided by Triton Inference Server.

## Triton Inference Server
Triton Inference Server enables teams to deploy AI models from various frameworks,
including deep learning and machine learning, across multiple devices and platforms,
ensuring optimized performance and flexibility. Its features, such as concurrent
model execution, dynamic batching, and sequence batching, allow for efficient and
scalable inference processing. Additionally, Triton's use of standard APIs and
protocols provide seamless integration with applications, enabling real-time and
batched inference, as well as monitoring and analytics capabilities, making it an
ideal solution for accelerating the data science pipeline and streamlining AI
development and deployment.

When the Triton Inference Server is launched, it will load the models that are found
in the model repository. This is nothing more than a directory structure with the
required files. When using the Python backend, as we are in this tutorial, it has
the following structure:

```
model_repository/
  <deployment1_name>/
    config.pbtxt # Specifies inputs/outputs + other deployment options
    1/           # Deployment version number
      model.py   # Recommend using unique name
  <deployment2_name>/
    config.pbtxt # Specifies inputs/outputs + other deployment options
    1/           # Deployment version number
      model.py   # Recommend using unique name
```
We will add the model.py & config.pbtxt files in the next tutorial.

## Setup
Before we can begin, we need to do some setup work. We will download both the Triton
Inference Server and the Triton Inference Server SDK containers.

### Pulling Containers
We will use the latest version of the docker containers (v24.05). We will pull the
Python & PyTorch backends and also the corresponding SDK container. Run the following
command to pull the needed containers.

```
$ docker pull nvcr.io/nvidia/tritonserver:24.05-pyt-python-py3
$ docker pull nvcr.io/nvidia/tritonserver:24.05-py3-sdk
```
### Creating Conda Environments
To enable each model to have the necessary libraries it needs, Triton Inference
Server uses [conda packs](https://conda.github.io/conda-pack/). The
[NVIDIA documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#creating-custom-execution-environments)
highlights the importance of setting `export PYTHONNOUSERSITE=True` before
calling conda-pack. In addition, the docs mention that the Python version in the
conda-pack **must** match the Python version in the container. You can check the
version of Python in the container with the following command.

```
$ docker run -it nvcr.io/nvidia/tritonserver:24.05-pyt-python-py3 /bin/bash
container:/opt/tritonserver# /usr/bin/python3 -V
container:/opt/tritonserver# exit
```
For this version of the container, you should see that it is using version 3.10.12.
For your convenience, there are environment.yml files for both of the models.

To create the fasttext-language-identification conda pack run the following code.

```
$ export PYTHONNOUSERSITE=True
$ cd model_repository/fasttext-language-identification
$ conda env create -f environment.yml
$ conda-pack -n fasttext-language-identification
```
This will place fasttext-language-identification.tar.gz in the
model_repository/fasttext-language-identification directory and will be read into
the Triton Inference Server container when it is launched.

Similarly, we need to make a conda pack for the SeamlessM4Tv2Large model.
```
$ cd ../seamless-m4t-v2-large
$ conda env create -f environment.yml
$ conda-pack -n seamless-m4t-v2-large
```
## Download Models
The two models that we will use are available in Huggingface. We want to download these
now so they are stored in your local Huggingface Hub cache, which will then be volume
mounted into the Triton Inference Server (see docker-compose.yaml). I'm sure there is
a better way, but I just open up a Python interpretter and pull the models following
the directions on their model pages. Huggingface gives some other ways to do [this](https://huggingface.co/docs/hub/en/models-downloading).

* [SeamlessM4Tv2Large](https://huggingface.co/facebook/seamless-m4t-v2-large)
* [FastText-Language-Identification](https://huggingface.co/facebook/fasttext-language-identification)


You can easily download models using the huggingface_hub library. Use the snapshot_download function with 
the repository ID, such as snapshot_download("facebook/fasttext-language-identification"). 
All downloaded files will be stored in a local cache folder for easy access.

```
$ cd ~/home/[USER_DIR]/.cache/huggingface/hub  #make sure your docker-compose.yaml has the correct path
$ python
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1")
```

## Next
In the first tutorial, we will create a basic translation service by defining
the config.pbtxt & model.py files for both models. We will hold off adding the
service level deployment, translate, until a later tutorial. This will be the
main deployment that we want clients to send their data to.
