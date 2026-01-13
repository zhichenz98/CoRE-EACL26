import json
import os
import threading

from transformers import LogitsProcessorList, TextStreamer

from logits_processor import TokenMappingProcessor
from utils.answer_extract import answer_extract

class MainThread(threading.Thread):
    """Drive main-model decoding while fusing assist-model signals via CoRE."""
    def __init__(self, main_tokenizer, main_model, token_map_list, information_dict, result_save_dir, ensemble_output_ids_queue, assist_score_queue_list, device, align_method, variant, task, early_stop_string_list, topK=10, entropy_thres=-1, tau=1, assist_tokenizer_list=None):
        super().__init__()
        self.device = device
        self.tokenizer = main_tokenizer
        self.model = main_model
        self.model_streamer = TextStreamer(self.tokenizer)
        self.token_map_list = token_map_list
        self.information_dict = information_dict
        self.result_save_dir = result_save_dir
        self.early_stop_string_list = early_stop_string_list
        self.ensemble_output_ids_queue = ensemble_output_ids_queue
        self.assist_score_queue_list = assist_score_queue_list
        self.topK = topK
        self.align_method = align_method
        self.variant=variant
        self.task=task
        self.entropy_thres=entropy_thres
        self.tau=tau
        self.assist_tokenizer_list = assist_tokenizer_list

    def run(self) -> None:
        # Build the logits processor that performs token mapping and consistency-based ensembling.
        processor_kwargs = {
            "ensemble_weight": self.information_dict["ensemble_weight"],
            "ensemble_output_ids_queue": self.ensemble_output_ids_queue,
            "assist_score_queue_list": self.assist_score_queue_list,
            "result_save_dir": self.result_save_dir,
            "main_tokenizer": self.tokenizer,
            "token_map_list": self.token_map_list,
            "device": self.device,
            "align_method": self.align_method,
            "variant": self.variant,
            "task": self.task,
            "early_stop_string_list": self.early_stop_string_list,
            "topK": self.topK,
            "entropy_thres": self.entropy_thres,
            "tau": self.tau,
            "assist_tokenizer_list": self.assist_tokenizer_list,
        }
        main_logits_processor_list = LogitsProcessorList(
            [TokenMappingProcessor(**processor_kwargs)])
        main_input = self.information_dict['main_model_input']
        max_new_tokens = self.information_dict['max_new_tokens']

        # Standard greedy decode; fusion happens inside TokenMappingProcessor.
        main_input_ids = self.tokenizer(main_input, return_tensors="pt",
                                              add_special_tokens=False).input_ids.to(self.device)
        generation_kwargs = {
            "input_ids": main_input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
        }

        # generation
        generate_ids = self.model.generate(**generation_kwargs, pad_token_id=self.tokenizer.eos_token_id, logits_processor=main_logits_processor_list, use_cache=False)
        text = self.tokenizer.decode(generate_ids[0])

        # Task-specific answer extraction for evaluation logging.
        result_process_parameter = self.information_dict['result_process_parameter']
        split_key_before_list = result_process_parameter["split_key_before"]
        split_key_behind_list = result_process_parameter["split_key_behind"]

        model_answer, prediction = answer_extract(text, self.information_dict['demon_count'], split_key_before_list,
                                                  split_key_behind_list)
        # print(self.information_dict['question'])
        # print(prediction.strip())
        model_answer_dict = {'answer': self.information_dict['answer'],
                             'prediction': prediction.strip(), 'main_model_input': main_input, 'all': text,
                             'model_answer': model_answer,
                             'question': self.information_dict['question']}

        if len(self.token_map_list) > 1:    # model ensemble
            result_file_path = os.path.join(self.result_save_dir, '{}-{}.jsonl'.format(self.align_method, self.variant))
        else:   # single model
            result_file_path = os.path.join(self.result_save_dir, 'single.jsonl')
        with open(result_file_path, 'a+', encoding='utf-8') as result_file:
            result_file.write(json.dumps(
                model_answer_dict, ensure_ascii=False) + '\n')
