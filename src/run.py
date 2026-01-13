from utils.instruction_generate import task_instruction_generate, demon_prompt_generate
from utils.model_load import load_model
from utils.load_token_map import load_token_map
from utils.safe_convert_ids_to_tokens import safe_convert_ids_to_tokens
import torch

import argparse
from main_thread import MainThread
from assist_thread import AssistThread

import json
import os
import logging
import queue
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')



# Entry point for CoRE test-time ensembling across general QA/code tasks.
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--task', default='PIQA', choices='NQ|GSM8K|PIQA|TriviaQA|ARC-c|SAMSum|HumanEval')
parser.add_argument('--main_model', default="InternLM7b", help='main model name')
parser.add_argument('--assist_model', type=lambda s: s.split(","), default=None, required=False, help='assist model name (split by comma)')
parser.add_argument('--variant', default="consist-rbf", type=str, choices="vanilla|consist-linear|consist-rbf|consist-power|consist-rec")
parser.add_argument('--align_method', default="unite", type=str, choices="unite|mined|gac|eva")
parser.add_argument('--topK', default=10, help='topK')
parser.add_argument('--device0', '-d0', default="cuda:0", type=str, required=False, help='device0')
parser.add_argument('--device1', '-d1', default="cuda:0", type=str, required=False, help='device1')
parser.add_argument('--device2', '-d2', default="cuda:0", type=str, required=False, help='device2')
parser.add_argument('--device3', '-d3', default="cuda:0", type=str, required=False, help='device3')
parser.add_argument('--device4', '-d4', default="cuda:0", type=str, required=False, help='device4')
parser.add_argument('--device5', '-d5', default="cuda:0", type=str, required=False, help='device5')
parser.add_argument('--device6', '-d6', default="cuda:0", type=str, required=False, help='device6')
parser.add_argument('--device7', '-d7', default="cuda:0", type=str, required=False, help='device7')
parser.add_argument('--device8', '-d8', default="cuda:0", type=str, required=False, help='device8')
parser.add_argument('--device_compute', '-dc', default="cuda:0", type=str, required=False,help='device_compute')
parser.add_argument('--result_save_dir', '-rsd', default=None,type=str, required=False, help='result_save_dir')
parser.add_argument('--run_mode', '-rm', default="test", choices=["dev", "test"], type=str, required=False, help='run_mode')
parser.add_argument('--ensemble_weight', '-ew', nargs='+', type=float,default=[1.0], help='ensemble_weight', required=False)
parser.add_argument('--entropy_thres', '-et', type=float,default=-1, help='entropy threshold', required=False)
parser.add_argument('--tau', '-tau', type=float, default=1, help='entropy weight temperature', required=False)

args = parser.parse_args()
start_time = time.time()
model_names = args.main_model
if args.assist_model is None:
    args.assist_model = []
for ele in args.assist_model:
    model_names += '+{}'.format(ele)
args.config = "./confs/{}/{}.json".format(args.task, model_names)

# Load task configuration and prompt templates.
with open(args.config, 'r', encoding='utf-8') as f:
    config_json = json.load(f)
model_paths = config_json["model_path"]
assist_count = len(model_paths) - 1

main_path = config_json["model_path"]["main_model_path"]
main_system_template = config_json["prompt_template"]["main_model_system_template"]

dev_file_path = config_json["file_path"]["dev_file_path"]
test_file_path = config_json["file_path"]["test_file_path"]
demon_file_path = config_json["file_path"]["demon_file_path"]

instruction = config_json["prompt_template"]["instruction"]
instruction_parameter = config_json["prompt_template"]["instruction_parameter"]
max_new_tokens = config_json["run_parameter"]["max_new_tokens"]

demon_parameter = config_json["prompt_template"]["demon_parameter"]
result_process_parameter = config_json["result_process_parameter"]

try:
    early_stop_string_list = result_process_parameter["early_stop_string_list"]
except:
    early_stop_string_list = None

# Resolve output directory for results and logs.
result_save_dir = args.result_save_dir
if result_save_dir is None:
    result_save_dir = "./res/{}/{}/{}".format(args.task, args.run_mode, model_names)
if os.path.isdir(result_save_dir):
    pass
else:
    os.makedirs(result_save_dir)

# Device placement for main model and assist models.
device0 = args.device0
device1 = args.device1
device2 = args.device2
device3 = args.device3
device4 = args.device4
device5 = args.device5
device6 = args.device6
device7 = args.device7
device8 = args.device8
device_compute = args.device_compute
device_list = [device0, device1, device2, device3,
                device4, device5, device6, device7, device8]
ensemble_weight = args.ensemble_weight
entropy_thres=args.entropy_thres

if len(model_paths) > 1:
    if ensemble_weight[0] != 1.0:
        assert len(ensemble_weight) == len(
            model_paths), "ensemble weight does not match # models"
        assert sum(
            ensemble_weight) == 1, "ensemble weight must have a sum of 1"
    else:
        ensemble_weight = [1.0 / len(model_paths)] * len(model_paths)   # default as average weight
    
# Process-level logging for reproducing ensemble decisions.
logging.basicConfig(filename=os.path.join(
    result_save_dir, '{}-{}.process.log'.format(args.align_method, args.variant)), level=logging.DEBUG)
logging.info(f'\n【config_json:】{config_json}')
logging.info(f'\n【result_save_dir:】{result_save_dir}')

# Load main/assist models and construct token-alignment maps.
main_model, main_tokenizer, main_streamer = load_model(main_path, "auto")
token_map_list = []
assist_list, assist_tokenizer_list = [], []
assist_streamer_list, assist_system_template_list = [], []
for assist_index in range(1, assist_count+1):
    assist_model, assist_tokenizer, assist_streamer = load_model(
        config_json["model_path"]["assist_model" + str(assist_index) + "_path"], "auto")
    assist_list.append(assist_model)
    assist_tokenizer_list.append(assist_tokenizer)
    assist_streamer_list.append(assist_streamer)
    token_map_list.append(load_token_map(args.main_model, args.assist_model[assist_index-1], main_model.config.vocab_size, assist_model.config.vocab_size, main_tokenizer, assist_tokenizer, main_model, assist_model, device_compute, args.align_method))
    assist_system_template_list.append(
        config_json["prompt_template"]["assist_model" + str(assist_index) + "_system_template"])

# Store results incrementally for long-running runs.
result_file_path = os.path.join(result_save_dir, f'.jsonl')
try:
    with open(result_file_path, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)
    start_index = line_count
except:
    start_index = 0

# Start sample-wise generation using a main thread + assist threads per step.
input_file_path = dev_file_path if args.run_mode == "dev" else test_file_path
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    try:
        demon_instruction, demon_count = demon_prompt_generate(
            demon_file_path, demon_parameter)
    except:
        demon_instruction = ""
        demon_count = 0
    contents = input_file.readlines()

    # Answer each question one by one with test-time ensembling.
    for index, line in enumerate(tqdm(contents[start_index:])):
        line = json.loads(line)

        # Q: xxx, A:
        task_instruction = task_instruction_generate(
            line, instruction_parameter)
        # Demons + Q: xxx, A:
        final_input_prompt = instruction + demon_instruction + task_instruction
        # <s> + Demons + Q: xxx, A:
        main_input = main_system_template.format(
            final_input_prompt)

        information_key_list = demon_parameter['key']
        information_dict = {}
        for key in information_key_list:
            information_dict[key] = line[key]
        information_dict['main_model_input'] = main_input
        information_dict['demon_count'] = demon_count
        information_dict['task_instruction'] = task_instruction
        information_dict['max_new_tokens'] = max_new_tokens
        information_dict['result_process_parameter'] = result_process_parameter
        information_dict['ensemble_weight'] = ensemble_weight

        ensemble_output_ids_queue = queue.Queue()
        main_input_ids_list = main_tokenizer.encode(main_input, add_special_tokens=False, return_tensors="pt").squeeze().to(main_model.device)

        assist_score_queue_list = []
        assist_input_list = []
        for assist_index in range(0, assist_count):
            assist_score_queue_list.append(queue.Queue())
            assist_input_list.append(assist_system_template_list[assist_index].format(final_input_prompt))

        # Main model thread starts (consumes assist logits via queues).
        main_thread = MainThread(
                            main_tokenizer=main_tokenizer,
                            main_model=main_model,
                            token_map_list = token_map_list,
                            information_dict=information_dict,
                            result_save_dir=result_save_dir,
                            ensemble_output_ids_queue=ensemble_output_ids_queue,
                            assist_score_queue_list=assist_score_queue_list,
                            device=device_compute,
                            early_stop_string_list=early_stop_string_list,
                            align_method = args.align_method,
                            task = args.task,
                            variant=args.variant,
                            entropy_thres=entropy_thres,
                            topK=args.topK,
                            tau=args.tau,
                            assist_tokenizer_list=assist_tokenizer_list
                            )
        main_thread.start()

        # Assist model threads run one-step decoding to provide logits for fusion.
        for i in range(max_new_tokens): # assist models generate up to max_new_tokens for each input sample

            assist_thread_list = []
            for assist_index in range(0, assist_count):
                assist_thread = AssistThread(model=assist_list[assist_index],
                                            model_tokenizer=assist_tokenizer_list[assist_index],
                                            assist_input=assist_input_list[assist_index],
                                            assist_score_queue=assist_score_queue_list[assist_index],
                                            device=device_list[assist_index+1],
                                            result_save_dir=result_save_dir
                                            )    # Assist thread generate one token per step
                assist_thread.start()
                assist_thread_list.append(assist_thread)

            for assist_thread in assist_thread_list:
                assist_thread.join()
            
            # End AssistThread, start MainThread for the fused next token.
            if max_new_tokens != 1: # obtain new ensemble token
                try:
                    ensemble_generate_next_id = ensemble_output_ids_queue.get(
                        block=True, timeout=4 + 0.0167 * max_new_tokens).to(device_compute)
                    logging.info(f'{i}, {safe_convert_ids_to_tokens(main_tokenizer, ensemble_generate_next_id)}')
                except:
                    break

                if ensemble_generate_next_id.item() == main_tokenizer.eos_token_id or ensemble_generate_next_id.item() == main_tokenizer.convert_tokens_to_ids("<|im_end|>"):
                        break
                
                main_input_ids_list = torch.cat((main_input_ids_list, ensemble_generate_next_id), dim=0)
                main_input = main_tokenizer.decode(main_input_ids_list, skip_special_tokens=True)
                for assist_index in range(len(assist_input_list)):
                    assist_input_list[assist_index] = main_input
                
        main_thread.join()

time_elapsed = time.time() - start_time  # 获得时间差
minutes = int(time_elapsed / 60)
seconds = int(time_elapsed % 60)
logging.info(f"\nTime taken: {minutes} min {seconds} sec")
print('Time taken: {} min {} sec'.format(minutes, seconds))
