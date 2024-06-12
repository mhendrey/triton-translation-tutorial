import json
import numpy as np

import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Perform translation using SeamlessM4T-large-v2's Text2Text"""

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        # Get TRANSLATED_TEXT configuration
        translated_text_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSLATED_TEXT"
        )
        # Convert Triton types to numpy types
        self.translated_text_dtype = pb_utils.triton_string_to_numpy(
            translated_text_config["data_type"]
        )

        # Use the GPU if available, otherwise use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            torch_dtype = torch.float32  # CPUs can't handle float16
        self.model = SeamlessM4Tv2ForTextToText.from_pretrained(
            "facebook/seamless-m4t-v2-large",
            device_map="auto",
            torch_dtype=torch_dtype,
            cache_dir="/hub",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            "facebook/seamless-m4t-v2-large",
            cache_dir="/hub",
            local_files_only=True,
        )

    def execute(self, requests: List) -> List:
        """
        Each request is sent by a client and represents appropriately chunked text
        for translation from a single document. The INPUT_TEXT is a 1-d array of
        bytes of variable length. The SRC_LANG and TGT_LANG inputs are 1-d array of
        bytes but have just one element in them.

        It is assumed that for each request that all the chunks of text to be
        translated are of the same source language and will be translated to the
        same target language. These will be done on the GPU at the same time.

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]

        Returns
        -------
        responses: List[pb_utils.InferenceResponse]
        """
        logger = pb_utils.Logger
        logger.log_warn(f"Seamless received {len(requests)} requests")
        responses = []
        for request in requests:
            # Get the input data as Triton Tensors
            try:
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
                src_lang_tt = pb_utils.get_input_tensor_by_name(request, "SRC_LANG")
                tgt_lang_tt = pb_utils.get_input_tensor_by_name(request, "TGT_LANG")
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{exc}", pb_utils.TritonError.INVALID_ARG
                    )
                )
                responses.append(response)
                continue

            # Convert TritonTensor -> numpy -> python str
            # NOTE: Triton converts your input string to bytes so you need to decode
            input_text = [b.decode("utf-8") for b in input_text_tt.as_numpy()]
            src_lang = src_lang_tt.as_numpy()[0].decode("utf-8")
            tgt_lang = tgt_lang_tt.as_numpy()[0].decode("utf-8")

            # Run through the model for translation
            ## Tokenize
            try:
                input_ids = self.processor(
                    text=input_text,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    return_tensors="pt",
                ).to(self.device)
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"processor threw:{exc}")
                )
                responses.append(response)
                continue

            ## Generate output tokens
            try:
                output_tokens = self.model.generate(
                    **input_ids,
                    tgt_lang=tgt_lang,
                    num_beams=5,
                    num_return_sequences=1,
                    max_new_tokens=3000,
                    no_repeat_ngram_size=3,
                )
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"model.generate threw: {exc}")
                )
                responses.append(response)
                continue

            ## Decode tokens to text
            try:
                translated_text = self.processor.batch_decode(
                    output_tokens, skip_special_tokens=True
                )
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("processor.batch_decode threw: {exc}")
                )
                responses.append(response)
                continue

            # Convert to TritonTensor & make the TritonInferenceResponse
            translated_text_tt = pb_utils.Tensor(
                "TRANSLATED_TEXT",
                np.array(translated_text, dtype=self.translated_text_dtype),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[translated_text_tt],
            )
            responses.append(inference_response)

        return responses
