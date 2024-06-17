from collections import defaultdict
import json
import logging
import numpy as np

import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
from typing import List

import triton_python_backend_utils as pb_utils

logger = logging.getLogger(__name__)


class SeamlessFix(SeamlessM4Tv2ForTextToText):
    def generate(
        self,
        input_ids=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    ):
        # prepare text_decoder_input_ids
        text_decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            batch_size = (
                len(input_ids)
                if input_ids is not None
                else len(kwargs.get("inputs_embeds"))
            )

            if hasattr(self.generation_config, "text_decoder_lang_to_code_id"):
                if isinstance(tgt_lang, str):
                    tgt_lang = [tgt_lang] * batch_size
                elif len(tgt_lang) != batch_size:
                    raise ValueError(
                        f"tgt_lang length, {len(tgt_lang)} != {batch_size} batch size"
                    )

                text_decoder_input_ids = []
                for tgt in tgt_lang:
                    # also accept __xxx__
                    tgt = tgt.replace("__", "")
                    if tgt not in self.generation_config.text_decoder_lang_to_code_id:
                        raise ValueError(
                            f"""`tgt_lang={tgt}` is not supported by this model. Please specify a `tgt_lang` in
                            {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}"""
                        )
                    # tgt_lang gets priority over decoder input ids
                    text_tgt_lang_id = (
                        self.generation_config.text_decoder_lang_to_code_id.get(tgt)
                    )
                    text_decoder_input_ids.append(text_tgt_lang_id)

                text_decoder_input_ids = (
                    torch.tensor(text_decoder_input_ids).reshape(-1, 1).to(self.device)
                )
            else:
                raise ValueError(
                    """This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps
                    the target language to the right token id. Make sure to load the right generation config."""
                )
        else:
            # only a warning, otherwise errors appear in the tests
            logger.warning(
                """You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get
                a correct generation, otherwise the generation will probably make no sense."""
            )

        return super(SeamlessM4Tv2ForTextToText, self).generate(
            input_ids,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=text_decoder_input_ids,
            **kwargs,
        )


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
        n_requests = len(requests)
        logger.log_warn(f"Seamless received {n_requests} requests")
        responses = [None] * n_requests
        # Because the model can only process a batch with all the same
        # src_lang & tgt_lang. We will batch them up using those as keys
        mini_batches_texts = defaultdict(list)
        mini_batches_sizes = defaultdict(list)
        mini_batches_requests = defaultdict(list)
        for i, request in enumerate(requests):
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
                responses[i] = response
                continue

            # Convert TritonTensor -> numpy -> python str
            # NOTE: Triton converts your input string to bytes so you need to decode
            input_text = [
                b.decode("utf-8") for b in input_text_tt.as_numpy().reshape(-1)
            ]
            src_lang = src_lang_tt.as_numpy().reshape(-1)[0].decode("utf-8")
            tgt_lang = tgt_lang_tt.as_numpy().reshape(-1)[0].decode("utf-8")

            mini_batches_texts[(src_lang, tgt_lang)] += input_text
            mini_batches_sizes[(src_lang, tgt_lang)].append(len(input_text))
            mini_batches_requests[(src_lang, tgt_lang)].append(i)

        # Loop through the src_lang/tgt_lang pairings of mini-batches
        for (src_lang, tgt_lang), input_text in mini_batches_texts.items():
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
                mini_batch_ptr = 0
                # We need to send an error to all requests in the mini-batch
                # Take care not to send the exc since it might have someone else's data
                for request_id, batch_size in zip(
                    mini_batches_requests[(src_lang, tgt_lang)],
                    mini_batches_sizes[(src_lang, tgt_lang)],
                ):
                    response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"processor threw an error for the batch you were in. You "
                            + f"sent. {input_text[mini_batch_ptr:batch_size]}. Check "
                            + "your input text for problems."
                        )
                    )
                    mini_batch_ptr += batch_size
                    responses[request_id] = response
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
                mini_batch_ptr = 0
                # We need to send an error to all requests in the mini-batch
                # Take care not to send the exc since it might have someone else's data
                for request_id, batch_size in zip(
                    mini_batches_requests[(src_lang, tgt_lang)],
                    mini_batches_sizes[(src_lang, tgt_lang)],
                ):
                    response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"model.generate threw an error for the batch you were in."
                            + f"You sent. {input_text[mini_batch_ptr:batch_size]}. "
                            + "Check your input text for problems."
                        )
                    )
                    mini_batch_ptr += batch_size
                    responses[request_id] = response
                continue

            ## Decode tokens to text
            try:
                translated_text = self.processor.batch_decode(
                    output_tokens, skip_special_tokens=True
                )
            except Exception as exc:
                mini_batch_ptr = 0
                # We need to send an error to all requests in the mini-batch
                # Take care not to send the exc since it might have someone else's data
                for request_id, batch_size in zip(
                    mini_batches_requests[(src_lang, tgt_lang)],
                    mini_batches_sizes[(src_lang, tgt_lang)],
                ):
                    response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"processor.batch_decode threw an error for the batch you were in. You "
                            + f"sent. {input_text[mini_batch_ptr:batch_size]}. Check "
                            + "your input text for problems."
                        )
                    )
                    mini_batch_ptr += batch_size
                    responses[request_id] = response
                continue

            # Split successfully translated mini-batch back into individual responses
            mini_batch_ptr = 0
            for request_id, batch_size in zip(
                mini_batches_requests[(src_lang, tgt_lang)],
                mini_batches_sizes[(src_lang, tgt_lang)],
            ):
                requested_translations = translated_text[
                    mini_batch_ptr : (mini_batch_ptr + batch_size)
                ]
                mini_batch_ptr += batch_size

                # Convert to TritonTensor & make the TritonInferenceResponse
                translated_text_tt = pb_utils.Tensor(
                    "TRANSLATED_TEXT",
                    np.array(
                        requested_translations, dtype=self.translated_text_dtype
                    ).reshape(-1, 1),
                )
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[translated_text_tt],
                )
                responses[request_id] = inference_response

        return responses
