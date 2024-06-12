import fasttext
from huggingface_hub import hf_hub_download
import json
import numpy as np
import re
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """This model uses fasttext to perform language identification"""

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        # Get output configs
        src_lang_config = pb_utils.get_output_config_by_name(model_config, "SRC_LANG")
        src_script_config = pb_utils.get_output_config_by_name(
            model_config, "SRC_SCRIPT"
        )

        # Convert Triton types to numpy types for output data
        self.src_lang_dtype = pb_utils.triton_string_to_numpy(
            src_lang_config["data_type"]
        )
        self.src_script_dtype = pb_utils.triton_string_to_numpy(
            src_script_config["data_type"]
        )

        """Load the model into CPU RAM"""
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification",
            filename="model.bin",
            cache_dir="/hub",
            local_files_only=True,
        )
        self.model = fasttext.load_model(model_path)
        self.REMOVE_NEWLINE = re.compile(r"\n")

    def execute(self, requests: List) -> List:
        """Predict the language id of the text provided in the request. Newlines are
        stripped since they throw an error. Only the top prediction is provided and
        irrespective of how low the confidence is.

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            Input must contain the INPUT_TEXT

        Returns
        -------
        List[pb_utils.InferenceResponse]
        """
        logger = pb_utils.Logger
        logger.log_warn(f"FastText received {len(requests)} requests")
        responses = []
        for request in requests:
            # Get INPUT_TEXT from request. This is a Triton Tensor
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
            # Convert Triton Tensor (TYPE_STRING) to numpy (dtype=np.object_)
            # Array has just one element (config.pbtxt has dims: [1])
            # TYPE_STRING is bytes when sending through a request. Decode to get str
            input_text = input_text_tt.as_numpy().reshape(-1)[0].decode("utf-8")
            # Replace newlines with ' '. FastText breaks on \n
            input_text_cleaned = self.REMOVE_NEWLINE.sub(" ", input_text)

            # Run through the model
            try:
                output_labels, _ = self.model.predict(input_text_cleaned, k=1)
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"{exc}")
                )
                responses.append(response)
                continue
            # Take just the first one because we used k = 1 in predict()
            # Returns '__label__<lang_id>_<script>', e.g., '__label__spa_Latn'
            src_lang, src_script = output_labels[0].replace("__label__", "").split("_")

            # Make Triton Inference Response
            src_lang_tt = pb_utils.Tensor(
                "SRC_LANG",
                np.array([src_lang], dtype=self.src_lang_dtype).reshape(-1, 1),
            )
            src_script_tt = pb_utils.Tensor(
                "SRC_SCRIPT",
                np.array([src_script], dtype=self.src_script_dtype).reshape(-1, 1),
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[src_lang_tt, src_script_tt],
            )
            responses.append(response)

        return responses
