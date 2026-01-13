import threading

from transformers import LogitsProcessorList

from logits_processor import AssistModelLogitsProcessor


class AssistThread(threading.Thread):
    """Run an assist model for a single-step decode and export logits to the ensemble."""
    def __init__(self, model, model_tokenizer, assist_input, assist_score_queue, device, result_save_dir):
        self.model = model
        self.model_tokenizer = model_tokenizer
        self.model_input = assist_input
        self.model_scores_queue = assist_score_queue
        self.device = device
        self.result_save_dir = result_save_dir
        super().__init__()

    def run(self) -> None:
        # Assist models produce one-step logits; the main thread aggregates them via queues.
        model_input_ids = self.model_tokenizer(self.model_input, return_tensors="pt",
                                               add_special_tokens=False).input_ids.to(self.device)
        assist_generation_kwargs = {
            "input_ids": model_input_ids,
            "max_new_tokens": 1,
            "do_sample": False,
            "num_beams": 1,
            "eos_token_id": self.model_tokenizer.eos_token_id,
            "bos_token_id": self.model_tokenizer.bos_token_id,
        }
        assist_logits_processor_list = LogitsProcessorList()
        assist_logits_processor_list.append(
            AssistModelLogitsProcessor(self.model_scores_queue, self.result_save_dir))
        # Use logits_processor to capture raw scores without changing decoding behavior.
        assist_generate_ids = self.model.generate(**assist_generation_kwargs, pad_token_id=self.model_tokenizer.eos_token_id, logits_processor=assist_logits_processor_list, use_cache=False)
