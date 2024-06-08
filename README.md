# Triton Inference Server Tutorial
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
required files. When using the Python backend, as we are in ths tutorial, it has
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
We will add the model.py & config.pbtxt files in the next tutorial (v1 branch).

## Setup
Before we can begin, we need to do some setup work. We will download both the Triton
Inference Server and the Triton Inference Server SDK containers.

### Pulling Containers
We will use the latest version of the docker containers (v24.04). We will pull the
Python & PyTorch backends and also the corresponding SDK container. Run the following
command to pull the needed containers.

```
$ docker pull nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3
$ docker pull nvcr.io/nvidia/tritonserver:24.04-py-sdk
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
$ docker run -it nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3 /bin/bash
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

## Next
In the next tutorial, v1 branch, we will create a basic translation service by defining
the config.pbtxt & model.py files for both models. In addition, we will add the
service level deployment, translate, that will be the main deployment that we want clients
to send their data to.

So let's switch to the v1 branch and get started.
