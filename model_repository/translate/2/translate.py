import asyncio
from collections import defaultdict
import json
import inspect
import numpy as np
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Service Level Deployment Package
    This handles things nicely for clients. Taking in 'strings' [really means bytes]
    and then handles the logic for using the language id model (if not specified)
    before passing on to translation model."""

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        # Get INPUT_TEXT configuration
        input_text_config = pb_utils.get_input_config_by_name(
            model_config, "INPUT_TEXT"
        )
        # Convert Triton types to numpy types
        self.input_text_dtype = pb_utils.triton_string_to_numpy(
            input_text_config["data_type"]
        )

        # Get TRANSLATED_TEXT configuration
        translated_text_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSLATED_TEXT"
        )
        # Convert Triton types to numpy types
        self.translated_text_dtype = pb_utils.triton_string_to_numpy(
            translated_text_config["data_type"]
        )

    async def execute(self, requests: List) -> List:
        """
        Each request is one document that a client has submitted for translation

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of request submitted by clients. In this simple example, this should
            have a length of just one since dynamic batching is not enabled.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            Each response is the translated document for a given client's request
        """
        logger = pb_utils.Logger
        n_requests = len(requests)
        logger.log_warn(f"`translate` received {n_requests} requests in dynamic batch")
        responses = [None] * n_requests
        is_ok = [True] * n_requests
        inference_response_awaits = []
        batch_chunk_ids = []
        results = defaultdict(dict)
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for batch_id, request in enumerate(requests):
            # Get INPUT_TEXT
            try:
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{exc}", pb_utils.TritonError.INVALID_ARG
                    )
                )
                responses[batch_id] = response
                is_ok[batch_id] = False
                continue

            # Get any optional parameters passed in.
            request_params = json.loads(request.parameters())
            src_lang = request_params.get("src_lang", None)
            tgt_lang = request_params.get("tgt_lang", "eng")
            tgt_lang_tt = pb_utils.Tensor(
                "TGT_LANG", np.array([tgt_lang], np.object_).reshape(-1, 1)
            )

            # If src_lang provide in request, then use it, else use FastText to get it
            if src_lang:
                src_lang_tt = pb_utils.Tensor(
                    "SRC_LANG", np.array([src_lang], np.object_).reshape(-1, 1)
                )

            # Chunk up the input_text_tt into pieces for translation
            for chunk_id, chunk_tt in enumerate(self.chunk_document(input_text_tt)):
                if src_lang is None:
                    infer_lang_id_request = pb_utils.InferenceRequest(
                        model_name="fasttext-language-identification",
                        requested_output_names=["SRC_LANG"],
                        inputs=[chunk_tt],
                    )
                    # Perform synchronous blocking inference request
                    infer_lang_id_response = infer_lang_id_request.exec()
                    if infer_lang_id_response.has_error():
                        err_msg = (
                            f"{chunk_id=:} had error: "
                            + f"{infer_lang_id_response.error().message()}"
                        )
                        response = pb_utils.InferenceResponse(
                            error=pb_utils.TritonError(err_msg)
                        )
                        responses[batch_id] = response
                        is_ok[batch_id] = False
                        break
                    src_lang_tt = pb_utils.get_output_tensor_by_name(
                        infer_lang_id_response, "SRC_LANG"
                    )
                infer_seamless_request = pb_utils.InferenceRequest(
                    model_name="seamless-m4t-v2-large",
                    requested_output_names=["TRANSLATED_TEXT"],
                    inputs=[chunk_tt, src_lang_tt, tgt_lang_tt],
                )
                # Perform asynchronous inference request
                inference_response_awaits.append(infer_seamless_request.async_exec())
                batch_chunk_ids.append((batch_id, chunk_id))

        # After submitting all the chunks for all the requests, wait for results
        inference_responses = await asyncio.gather(*inference_response_awaits)
        for infer_seamless_response, (batch_id, chunk_id) in zip(
            inference_responses, batch_chunk_ids
        ):
            if infer_seamless_response.has_error() and responses[batch_id] is None:
                err_msg = (
                    f"{chunk_id=:} had error: "
                    + f"{infer_seamless_response.error().message()}"
                )
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(err_msg)
                )
                responses[batch_id] = response
                is_ok[batch_id] = False
            else:
                translated_chunk_np = (
                    pb_utils.get_output_tensor_by_name(
                        infer_seamless_response, "TRANSLATED_TEXT"
                    )
                    .as_numpy()
                    .reshape(-1)
                )
                translated_chunk = translated_chunk_np[0].decode("utf-8")
                results[batch_id][chunk_id] = translated_chunk

        for batch_id in sorted(results):
            if is_ok[batch_id]:
                result = results[batch_id]
                translated_chunks = [result[chunk_id] for chunk_id in sorted(result)]
                translated_doc = " ".join(translated_chunks)
                translated_doc_tt = pb_utils.Tensor(
                    "TRANSLATED_TEXT",
                    np.array([translated_doc], dtype=self.translated_text_dtype),
                )
                # Create the response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[translated_doc_tt]
                )
                responses[batch_id] = inference_response

        return responses

    def chunk_document(self, doc_text_tt, char_encoding: str = "utf-8"):
        """
        Split input document into appropriately sized chunks that can be sent to the
        translation model. In this simple example, we will naively split on "." as
        a very rough approximation of sentences. More care should be done here for
        a system to be deployed into production.

        Parameters
        ----------
        doc_text : pb_utils.Tensor
            Input tensor sent in the request from the client

        Returns
        -------
        chunks : pb_utils.Tensor
            Resultant array of input document split into sentences
        """
        doc_text = doc_text_tt.as_numpy().reshape(-1)[0].decode(char_encoding)
        for chunk in doc_text.split("."):
            if chunk:
                chunk = f"{chunk}."  # put . back on
                chunk_tt = pb_utils.Tensor(
                    "INPUT_TEXT",
                    np.array([chunk], dtype=self.translated_text_dtype).reshape(-1, 1),
                )
                yield chunk_tt

    def combine_translated_chunks(
        self, translated_chunks_tt, char_encoding: str = "utf-8"
    ):
        """
        Take translated chunks (Triton Tensor) sent back from the translation model
        and combine them into a single string. In this simple example, we naively just
        join with " ". Again, more sophisticated approaches could be coded here.

        Parameters
        ----------
        translated_chunks : pb_utils.Tensor
            1-d Tensor of the translated chunks

        Returns
        -------
        translated_doc : pb_utils.Tensor
            translated document
        """
        translated_chunks = [
            b.decode(char_encoding) for b in translated_chunks_tt.as_numpy()
        ]
        translated_doc = " ".join(translated_chunks)
        translated_doc_tt = pb_utils.Tensor(
            "TRANSLATED_TEXT",
            np.array([translated_doc], dtype=self.translated_text_dtype),
        )
        return translated_doc_tt
