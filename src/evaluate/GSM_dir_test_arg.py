import json
import os
import os.path
import re
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import os
import os.path
import re
import sys
import argparse

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_last_num(text: str):
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return str(num_str)
    else:
        return None
    
def extract_first_num(text: str):
    text = re.sub(r"(\d),(\d)", r"\1\2", text)  # 处理形如 123,456
    res = re.search(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if res:
        return res.group(1)  # 提取第一个匹配的数字
    return None


def is_correct(model_answer, answer):
    return model_answer == answer


def result_write(result_path, sys_file_name, num_correct, num_total, accuracy, write=False):
    with open(os.path.join(result_path, 'EM_accuracy.jsonl'), 'a+', encoding='utf-8') as result_file:
        dict = {}
        dict['accuracy'] = '{:.2f}'.format(accuracy)
        dict['num_correct'] = num_correct
        dict['num_total'] = num_total
        if write:
            match = re.search(r'lr(.*?)learning_epochs_nums(.*)', sys_file_name)
            lr, anchor_point_count, learning_epochs_nums = match.groups()
            dict['learning_rate'] = lr.strip('_')
            dict['sys_file_path'] = os.path.join(result_path, sys_file_name)
            dict['learning_epochs_nums'] = learning_epochs_nums.strip('.jsonl')

        result_file.write(json.dumps(dict, ensure_ascii=False) + '\n')


def find_files_with_suffix(folder_path, suffix):
    # 使用os模块获取文件夹中所有文件的路径
    all_files = os.listdir(folder_path)
    # 筛选以指定后缀名结尾的文件
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files


# def extract_last_num(text: str):
#     text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
#     res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
#     if len(res) > 0:
#         num_str = res[-1][0]
#         return str(num_str)
#     else:
#         return None


# def is_correct(model_answer, answer):
#     return model_answer == answer


# def result_write(result_path, sys_file_name, num_correct, num_total, accuracy):
#     with open(os.path.join(result_path, 'EM_accuracy.jsonl'), 'a+', encoding='utf-8') as result_file:
#         dict = {}
#         match = re.search(r'lr(.*?)learning_epochs_nums(.*)', sys_file_name)
#         dict['accuracy'] = '{:.2f}'.format(accuracy)
#         dict['num_correct'] = num_correct
#         dict['num_total'] = num_total
#         dict['sys_file_path'] = os.path.join(result_path, sys_file_name)

#         result_file.write(json.dumps(dict, ensure_ascii=False) + '\n')


# def find_files_with_suffix(folder_path, suffix):
#     # 使用os模块获取文件夹中所有文件的路径
#     all_files = os.listdir(folder_path)
#     # 筛选以指定后缀名结尾的文件
#     filtered_files = [file for file in all_files if file.endswith(suffix)]
#     return filtered_files


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM ensemble results.")

    parser.add_argument("--task", type=str, default="GSM8K",
                        choices=["GSM8K"],
                        help="Evaluation task")
    parser.add_argument("--run_mode", type=str, default="dev",
                        choices=["dev", "test"],
                        help="Run mode: dev or test")
    parser.add_argument("--main_model", type=str, default="Llama2",
                        help="Main model name")
    parser.add_argument("--align_method", type=str, default="our",
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
    args.task = "GSM8K"

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
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.readlines()
        ref_file_dict = {}
        sys_file_dict = {}
        correct_count = 0
        for line in contents:
            json_obj = json.loads(line)
            json_obj['question'] = json_obj['question'].strip()

            # json_obj['prediction'] = extract_last_num(json_obj['all'])
            # try:    # only one answer
            #     json_obj['answer'] = extract_last_num(json_obj['answer'])
            #     if is_correct(json_obj['prediction'], json_obj['answer']):
            #         correct_count += 1
            # except:
            #     for ele in json_obj['answer']:
            #         answer = extract_last_num(ele)
            #         if is_correct(json_obj['prediction'], answer):
            #             correct_count += 1
            #             break
            json_obj['prediction'] = extract_first_num(json_obj['prediction'])
            json_obj['answer'] = extract_last_num(json_obj['answer'])
            if is_correct(json_obj['prediction'], json_obj['answer']):
                correct_count += 1

    accuracy = correct_count / len(contents)
    num_correct = correct_count
    num_total = len(contents)
    models = [ele for ele in args.assist_model]
    models = [args.main_model] + models
    # print(num_correct)
    # print(num_total)
    # print('{:.2f}'.format(accuracy * 100))
    print('Task: {}, Models: {}'.format(args.task, models))
    print('{:.2f}'.format(accuracy * 100))
    result_write(result_file_dir, file_path, num_correct, num_total, accuracy)


if __name__ == "__main__":
    main()
