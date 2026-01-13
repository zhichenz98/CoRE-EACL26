import json
import os
import sys
import nltk
import torch
from rouge import Rouge
from bert_score import score
import argparse
import warnings
warnings.filterwarnings("ignore")

def load_jsonl(file_path):
    """
    读取 JSONL 文件并解析 references 和 predictions
    """
    references, predictions = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            references.append(json_obj["answer"])
            predictions.append(json_obj["prediction"])
            # print("answers:", [json_obj["answer"]], "pred:", [json_obj["prediction"]])
    return references, predictions

def compute_bertscore(references, predictions):
    """
    计算 BERTScore 分数
    """
    P, R, F1 = score(predictions, references, lang="en")
    avg_f1 = F1.mean().item() * 100  # 转换为百分比
    return avg_f1

def compute_rouge(references, predictions):
    """
    计算 ROUGE 分数
    """
    rouge = Rouge()
    predictions = [p if p.strip() else "empty" for p in predictions]
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores

def find_files_with_suffix(folder_path, suffix):
    # 使用 os 模块获取文件夹中所有文件的路径
    all_files = os.listdir(folder_path)
    # 筛选以指定后缀名结尾的文件
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files

# 示例用法：
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM ensemble results.")

    parser.add_argument("--task", type=str, default="SAMSum",
                        choices=["SAMSum"],
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
    references, predictions = load_jsonl(file_path)
    print()
    print(f"{len(references)}/{len(predictions)}", file_path)
    # # 计算 BERTScore 分数
    # average_bertscore = compute_bertscore(references, predictions)
    # print(f"Average BERTScore F1: {average_bertscore:.2f}%")
    
    # 计算 ROUGE 分数
    rouge_scores = compute_rouge(references, predictions)
    print(f"ROUGE Scores: {rouge_scores}")