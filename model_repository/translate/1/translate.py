import json
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
        # Get TRANSLATED_TEXT configuration
        output_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSLATED_TEXT"
        )
        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests: List) -> List:
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

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT_TEXT
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

            # Get any optional parameters passed in.
            request_params = json.loads(request.parameters())
            src_lang = request_params.get("src_lang", None)
            tgt_lang = request_params.get("tgt_lang", "eng")
            tgt_lang_tt = pb_utils.Tensor("TGT_LANG", np.array([tgt_lang], np.object_))

            # If the lang_id isn't passed in, then run language id model to set it
            if not src_lang:
                # Create inference request object
                infer_lang_id_request = pb_utils.InferenceRequest(
                    model_name="fasttext-language-identification",
                    requested_output_names=["SRC_LANG"],
                    inputs=[input_text_tt],
                )

                # Peform synchronous blocking inference request
                infer_lang_id_response = infer_lang_id_request.exec()
                if infer_lang_id_response.has_error():
                    raise pb_utils.TritonModelException(
                        infer_lang_id_response.error().message()
                    )

                # Get the lang_id
                src_lang_tt = pb_utils.get_output_tensor_by_name(
                    infer_lang_id_response, "SRC_LANG"
                )
            else:
                src_lang_tt = pb_utils.Tensor(
                    "SRC_LANG", np.array([src_lang], np.object_)
                )

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
                raise pb_utils.TritonModelException(
                    infer_seamless_response.error().message()
                )

            # Get translated chunks
            translated_chunks_tt = pb_utils.get_output_tensor_by_name(
                infer_seamless_response, "TRANSLATED_TEXT"
            )
            # Combine translated chunks
            translated_doc_tt = self.combine_translated_chunks(translated_chunks_tt)

            # Create the response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[translated_doc_tt]
            )
            responses.append(inference_response)
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
        doc_text = doc_text_tt.as_numpy()[0].decode(char_encoding)
        chunks = []
        for chunk in doc_text.split("."):
            if chunk:
                chunks.append(f"{chunk}.")  # put . back on

        chunks_tt = pb_utils.Tensor("INPUT_TEXT", np.array(chunks, dtype=np.object_))
        return chunks_tt

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
            "TRANSLATED_TEXT", np.array([translated_doc], np.object_)
        )
        return translated_doc_tt
