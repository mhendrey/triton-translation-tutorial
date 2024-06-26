import itertools
import json
import numpy as np
import torch
from typing import List

from seamless_fix import SeamlessM4TProcessorMulti, SeamlessM4Tv2ForTextToTextMulti
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

    def execute(self, requests: List) -> List:
        """
        Each request is sent by a client and represents appropriately chunked text
        for translation. The INPUT_TEXT is a 1-d array of with one element of bytes.
        The SRC_LANG and TGT_LANG inputs are 1-d array of bytes but have just one
        element in them.

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]

        Returns
        -------
        responses: List[pb_utils.InferenceResponse]
        """
        responses = [None] * len(requests)
        valid_requests = []
        batch_input_text = []
        batch_src_lang = []
        batch_tgt_lang = []
        for i, request in enumerate(requests):
            try:
                # Get the input data as Triton Tensors
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
                src_lang_tt = pb_utils.get_input_tensor_by_name(request, "SRC_LANG")
                tgt_lang_tt = pb_utils.get_input_tensor_by_name(request, "TGT_LANG")

                # Convert TritonTensor -> numpy -> python str
                # NOTE: Triton converts your input string to bytes so you need to decode
                input_text = [
                    b.decode("utf-8") for b in input_text_tt.as_numpy().reshape(-1)
                ]
                src_lang = [
                    b.decode("utf-8") for b in src_lang_tt.as_numpy().reshape(-1)
                ]
                tgt_lang = [
                    b.decode("utf-8") for b in tgt_lang_tt.as_numpy().reshape(-1)
                ]

                batch_input_text.append(input_text)
                batch_src_lang.append(src_lang)
                batch_tgt_lang.append(tgt_lang)
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{exc}", pb_utils.TritonError.INVALID_ARG
                    )
                )
                responses[i] = response
                continue
            else:
                valid_requests.append(i)

        input_texts = list(itertools.chain.from_iterable(batch_input_text))
        src_langs = list(itertools.chain.from_iterable(batch_src_lang))
        tgt_langs = list(itertools.chain.from_iterable(batch_tgt_lang))
        # Run through the model for translation
        ## Tokenize
        try:
            input_ids = self.processor(
                text=input_texts,
                src_lang=src_langs,
                return_tensors="pt",
            ).to(self.device)
        except Exception as exc:
            # Error with the batch. Be careful error msg doesn't cross
            # contaminate user data
            for i in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"processor threw error tokenizing the batch"
                    )
                )
                responses[i] = response
            return responses

        ## Generate output tokens
        try:
            output_tokens = self.model.generate(
                **input_ids,
                tgt_lang=tgt_langs,
                num_beams=3,
                num_return_sequences=1,
                max_new_tokens=3000,
                no_repeat_ngram_size=3,
            )
        except Exception as exc:
            for i in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"model.generate threw error on batch")
                )
                responses[i] = response
            return responses

        ## Decode tokens to text
        try:
            translated_texts = self.processor.batch_decode(
                output_tokens, skip_special_tokens=True
            )
        except Exception as exc:
            for i in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("processor.batch_decode threw on batch")
                )
                responses[i] = response
            return responses

        for i, translated_text in zip(valid_requests, translated_texts):
            # Convert to TritonTensor & make the TritonInferenceResponse
            translated_text_tt = pb_utils.Tensor(
                "TRANSLATED_TEXT",
                np.array(translated_text, dtype=self.translated_text_dtype).reshape(
                    -1, 1
                ),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[translated_text_tt],
            )
            responses[i] = inference_response

        return responses
