import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluate.utils.NQ_evaluate_predictions import NQ_evaluate

import json
import os.path
import re


def result_write(result_path, sys_file_name, num_correct, num_total, accuracy):
    with open(os.path.join(result_path, 'EM_accuracy_all.jsonl'), 'a+', encoding='utf-8') as result_file:
        dict = {}

        match = re.search(r'lr(.*?)learning_epochs_nums(.*)', sys_file_name)
        dict['accuracy'] = '{:.2f}'.format(accuracy * 100)
        dict['num_total'] = num_total
        dict['num_correct'] = num_correct
        dict['sys_file_path'] = os.path.join(result_path, sys_file_name)

        result_file.write(json.dumps(dict, ensure_ascii=False) + '\n')


def find_files_with_suffix(folder_path, suffix):
    # 使用os模块获取文件夹中所有文件的路径
    all_files = os.listdir(folder_path)
    # 筛选以指定后缀名结尾的文件
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM ensemble results.")

    parser.add_argument("--task", type=str, default="NQ",
                        choices=["NQ", "PIQA", "MMLU", "TriviaQA", "ARC-c"],
                        help="Evaluation task")
    parser.add_argument("--run_mode", type=str, default="dev",
                        choices=["dev", "test"],
                        help="Run mode: dev or test")
    parser.add_argument("--main_model", type=str, default="Llama2",
                        help="Main model name")
    parser.add_argument("--align_method", type=str, default="unite",
                        choices=["unite", "mined", "gac", "eva"],
                        help="Alignment method")
    parser.add_argument("--assist_model", type=str, nargs="+",
                        default=[],
                        help="List of assist models")
    parser.add_argument("--variant", type=str, default="vanilla",
                        choices=["vanilla", "consist-linear", "consist-rbf", "consist-power", "consist-rec"],
                        help="Variant type")
    parser.add_argument("--res_dir", type=str,
                        default="./res/",
                        help="Base directory for results")

    args = parser.parse_args()

    # Build result file directory
    result_file_dir = os.path.join(args.res_dir, args.task, args.run_mode, args.main_model)
    for ele in args.assist_model:
        result_file_dir = "+".join((result_file_dir, ele))

    # Build jsonl file path
    if len(args.assist_model) == 0: # single model
        jsonl_files = os.path.join(
            result_file_dir,
            "single.jsonl"
        )
    else:   # model ensemble
        jsonl_files = os.path.join(
            result_file_dir,
            "{}-{}.jsonl".format(args.align_method, args.variant)
        )
    file_path = jsonl_files

    # Run evaluation
    is_choice = args.task in ["PIQA", "ARC-c"]
    num_correct, num_total, accuracy = NQ_evaluate(file_path, file_path, is_choice=is_choice)

    models = [args.main_model] + args.assist_model

    print("Task: {}, Models: {}".format(args.task, models))
    print("{:.2f}".format(accuracy * 100))

    result_write(result_file_dir, file_path, num_correct, num_total, accuracy)


if __name__ == "__main__":
    main()
