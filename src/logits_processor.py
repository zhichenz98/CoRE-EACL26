import json
import math
import os
import queue

import torch
from torch import nn
from transformers import LogitsProcessor
from utils.safe_convert_ids_to_tokens import safe_convert_ids_to_tokens

# Global switches for ablation/debug runs of the CoRE ensemble pipeline.
transit_probs = True    # do softmax before transition
entropy_routing = False # ensemble only if entropy is high
entropy_weighting = False   # entropy-based ensemble weight
check_substring = True  # check ensemble token is substring/superstring of main token
debug = False
rbf_tau = 1.0
eps = 1e-12



def compute_consist_score(prob_list, method="consist-rbf", task='NQ', **kwargs):
    """
    Compute token consistency scores for a list of probability distributions.

    Args:
        prob_list (list of torch.Tensor): each tensor is shape (1, d), probability distribution
        method (str): one of ["linear", "power", "exp", "reciprocal", "sigmoid", "pmi"]
        kwargs: extra parameters for different methods
            - power: alpha > 0
            - exp: beta > 0
            - reciprocal: gamma > 0
            - sigmoid: k > 0

    Returns:
        torch.Tensor: shape (N, d) consistency scores, same shape as prob_list stacked
    """
    # Stack into (N, d)
    probs = torch.cat(prob_list, dim=0)   # (N,d)
    N, d = probs.shape

    # Compute barycenter distribution for token-level agreement.
    p_star = probs.mean(dim=0, keepdim=True)  # (1,d)
    diff = torch.abs(probs - p_star)  # (N,d)

    # Mask avoids scores on impossible tokens (zero probability across models).
    p_mask = ((probs + probs[0].unsqueeze(0)) > eps).float()
    
    if method == "consist-linear":
        token_scores = p_mask * (1 - diff)

    elif method == "consist-power":
        alpha = kwargs.get("alpha", 5.0)
        token_scores = p_mask * (1 - diff).clamp(min=0) ** alpha

    elif method == "consist-rbf":
        beta = kwargs.get("beta", 2.0)
        token_scores = p_mask * (torch.exp(-beta * diff) - torch.exp(-torch.tensor(beta))) / (1 - torch.exp(-torch.tensor(beta)))

    elif method == "consist-rec":
        gamma = kwargs.get("gamma", 1.0)
        tmp = 1 / (1 + gamma * diff)
        token_scores = p_mask * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-8) 

    elif method == "consist-sig":
        k = kwargs.get("k", 5.0)
        token_scores = p_mask * (1 - torch.sigmoid(k * (diff - 0.5)))

    else:
        raise ValueError(f"Unknown method {method}")

    # Normalize per-model token scores to keep comparable scale across variants.
    nonzero_count = (token_scores != 0).sum(dim=1, keepdim=True).float()
    token_scores_sum = token_scores.sum(dim=1, keepdim=True)
    token_scores = token_scores / token_scores_sum * nonzero_count
    
    p_ent = row_entropy(probs)

    # Map token-level agreement to model-level weights per task type.
    if task in ['NQ', 'TriviaQA', 'GSM8K', 'SAMSum']:
        model_scores = token_scores_sum / (p_ent + eps)  # model score = h(prob) * sum token_score(x)
    
    elif task in ['ARC-c', 'MMLU', 'PIQA']:
        model_scores = 1 / (p_ent + eps)  # model score = h(prob) * sum token_score(x)
        
    else:
        raise ValueError('Unrecognized task type')
    
    # Clip main model weight to preserve a strong anchor distribution.
    model_scores[0] = model_scores[0].clamp_min(model_scores.sum() - model_scores[0])
    model_scores = (model_scores / model_scores.sum()).squeeze()
    
    return token_scores, model_scores



def row_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute per-row entropy over probability distributions."""
    row_sum = x.sum(dim=1, keepdim=True).clamp_min(eps)
    p = (x / row_sum).clamp_min(eps)
    logp = torch.log(p)
    h = -(p * logp).sum(dim=1, keepdim=True)
    return h


class TokenMappingProcessor(LogitsProcessor):
    """Align assist-model token distributions into main vocab and apply CoRE ensembling."""
    def __init__(self, ensemble_weight,
                 ensemble_output_ids_queue,
                 assist_score_queue_list, result_save_dir, main_tokenizer,
                 token_map_list, device, align_method, variant, task, early_stop_string_list=None, topK=10, entropy_thres=-1, tau=1, assist_tokenizer_list=None):
        self.assist_score_queue_list = assist_score_queue_list
        self.ensemble_weight = ensemble_weight
        self.ensemble_output_ids_queue = ensemble_output_ids_queue
        self.result_save_dir = result_save_dir
        self.main_tokenizer = main_tokenizer
        self.token_map_list = token_map_list
        self.device = device
        self.early_stop_string_list = early_stop_string_list
        self.topK = topK          
        self.align_method = align_method
        self.variant = variant
        self.task = task
        if entropy_thres < 0:  # give a valid entropy threshold
            entropy_thres = math.log(topK) / 2
        self.entropy_thres = entropy_thres
        self.tau = tau  # temperature for entropy weight
        self.assist_tokenizer_list = assist_tokenizer_list
        

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Collect assist-model logits from the previous step; fallback to main-only on timeout.
        ensemble_process_file_path = os.path.join(
            self.result_save_dir, '{}-{}.log'.format(self.align_method, self.variant))
        main_only_flag = False
        json_object = {}

        # obtain new logits for assist models
        assist_logits_list = []
        for index, queue_instance in enumerate(self.assist_score_queue_list):
            try:
                value = queue_instance.get(block=True, timeout=5)
                assist_logits_list.append(value)

            except queue.Empty:
                print(f"aux model{index}【not received】\n")
                assist_logits_list.append(None)
                main_only_flag = True
        if len(assist_logits_list) == 0:  # no assist scores obtained
            main_only_flag = True
        if torch.argmax(scores).item() == self.main_tokenizer.eos_token_id:
            main_only_flag = True

        if self.early_stop_string_list is not None:  # compare whether new token is early stop token
            for early_stop_string in self.early_stop_string_list:
                early_stop_token = self.main_tokenizer(early_stop_string, return_tensors="pt",
                                                       add_special_tokens=False).input_ids.tolist()[0]
                early_stop_token = early_stop_token[1:] if len(early_stop_token) > 1 else early_stop_token

                last_token_count = len(early_stop_token)

                last_token_ids = input_ids.tolist()[0][-last_token_count:]
                if last_token_ids == early_stop_token:
                    scores[:, self.main_tokenizer.eos_token_id] = float('inf')
                    main_only_flag = True
                    
        with torch.no_grad():
            # Restrict to top-K to reduce mapping cost and align with the paper's token-level ensembling.
            main_logits = scores.to(torch.float32).to(self.device)   # main model probs
            main_logits_values, main_logits_indices = torch.topk(main_logits, k=self.topK, dim=-1)    # obtain top-K main model output
            main_topK_probs = torch.full_like(main_logits, float("-inf"))
            main_topK_probs = main_topK_probs.scatter_(-1, main_logits_indices, main_logits_values)
            main_topK_probs = nn.functional.softmax(main_topK_probs, dim=-1)
                
        probs_list = []

        if not main_only_flag:    # when stop criterion is not met
            if self.variant == 'vanilla':
                with torch.no_grad():
                    
                    probs_list.append(main_topK_probs)
                    # assist model output
                    assist_probs_list = []
                    for assist_index, assist_logits in enumerate(assist_logits_list):
                        assist_logits_values, assist_logits_indices = torch.topk(assist_logits, k=self.topK)
                        assist_topK_probs = torch.full_like(assist_logits, float('-inf'))
                        assist_topK_probs = assist_topK_probs.scatter_(-1, assist_logits_indices, assist_logits_values)
                        assist_topK_probs = nn.functional.softmax(assist_topK_probs, dim=-1)
                        assist_probs_list.append(assist_topK_probs)
                        # Align assist vocab to main vocab before ensembling.
                        assist_token_map = self.token_map_list[assist_index][1]
                        assist_to_main_probs =  torch.sparse.mm(assist_topK_probs.to_sparse(), assist_token_map.to_sparse()).to_dense()
                        probs_list.append(assist_to_main_probs)

                    # Vanilla ensemble uses uniform weights when no model-level signal is requested.
                    self.ensemble_weight = torch.ones_like(torch.tensor(self.ensemble_weight))
                    self.ensemble_weight = torch.ones_like(self.ensemble_weight) / self.ensemble_weight.sum()
                    
                    ensemble_probs = probs_list[0] * self.ensemble_weight[0]
                    for assist_index in range(1, len(self.ensemble_weight)):
                        ensemble_probs += probs_list[assist_index] * self.ensemble_weight[assist_index]
                        
                    # Resolve token-level disagreements; keep main token on substring conflicts.
                    next_tokens_id = torch.argmax(ensemble_probs, dim=-1)
                    main_tokens_id = torch.argmax(probs_list[0], dim=-1)
                    ensemble_token = safe_convert_ids_to_tokens(self.main_tokenizer, next_tokens_id)[0]
                    main_token = safe_convert_ids_to_tokens(self.main_tokenizer,main_tokens_id)[0]
                    
                    if self.task in ['NQ', 'TriviaQA', 'GSM8K', 'SAMSum', 'HumanEval']: # check substring
                        if check_substring and (ensemble_token.startswith(main_token) or main_token.startswith(ensemble_token)):
                            next_tokens_id = main_tokens_id
                            ensemble_token = main_token
                            ensemble_probs = main_topK_probs
                        
            elif 'consist' in self.variant:
                with torch.no_grad():
                    probs_list.append(main_topK_probs)
                    # assist model output
                    assist_probs_list = []
                    for assist_index, assist_logits in enumerate(assist_logits_list):
                        assist_logits_values, assist_logits_indices = torch.topk(assist_logits, k=self.topK)
                        assist_topK_probs = torch.full_like(assist_logits, float('-inf'))
                        assist_topK_probs = assist_topK_probs.scatter_(-1, assist_logits_indices, assist_logits_values)
                        assist_topK_probs = nn.functional.softmax(assist_topK_probs, dim=-1)
                        assist_probs_list.append(assist_topK_probs)
                        # Align assist vocab to main vocab before ensembling.
                        assist_token_map = self.token_map_list[assist_index][1]
                        assist_to_main_probs =  torch.sparse.mm(assist_topK_probs.to_sparse(), assist_token_map.to_sparse()).to_dense()
                        probs_list.append(assist_to_main_probs)

                # Compute token-level consistency and derive model-level weights.
                token_scores, self.ensemble_weight = compute_consist_score(probs_list, self.variant, self.task)
                
                ensemble_probs = probs_list[0] * self.ensemble_weight[0]
                for assist_index in range(1, len(self.ensemble_weight)):
                    if self.task in ['PIQA', 'ARC-c', 'MMLU']:
                        ensemble_probs += probs_list[assist_index] * self.ensemble_weight[assist_index]
                    elif self.task in ['NQ', 'TriviaQA', 'GSM8K', 'SAMSum', 'HumanEval']:
                        ensemble_probs += probs_list[assist_index] * self.ensemble_weight[assist_index] * token_scores[assist_index]
                    else:
                        raise ValueError(f'Unknown task {self.task}!')

                # Resolve token-level disagreements; keep main token on substring conflicts.
                next_tokens_id = torch.argmax(ensemble_probs, dim=-1)
                main_tokens_id = torch.argmax(main_topK_probs, dim=-1)
                ensemble_token = safe_convert_ids_to_tokens(self.main_tokenizer, next_tokens_id)[0]
                main_token = safe_convert_ids_to_tokens(self.main_tokenizer,main_tokens_id)[0]
                if self.task in ['NQ', 'TriviaQA', 'GSM8K', 'SAMSum', 'HumanEval']: # check substring
                    if check_substring and (ensemble_token.startswith(main_token) or main_token.startswith(ensemble_token)):
                        next_tokens_id = main_tokens_id
                        ensemble_token = main_token
                        ensemble_probs = main_topK_probs
            
            else:
                raise ValueError('Invalid variant parameter!')
            
            if debug:
                json_object[f'main_tokens'] = (main_token, '{:.2f}, {:.2f}'.format(main_topK_probs[0,main_tokens_id].item(), self.ensemble_weight[0]))
                json_object[f'ensemble_token'] = (ensemble_token, '{:.2f}'.format(ensemble_probs[0,next_tokens_id].item()))
                
                for assist_index in range(len(self.assist_tokenizer_list)):
                    assist_tokens_id = torch.argmax(assist_probs_list[assist_index], dim=-1)
                    json_object['assist{}_tokens'.format(assist_index+1)] = (safe_convert_ids_to_tokens(self.assist_tokenizer_list[assist_index],assist_tokens_id), '{:.2f}, {:.2f}'.format(assist_probs_list[assist_index][0,assist_tokens_id].item(), self.ensemble_weight[assist_index+1]))
                
                with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                    process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

            self.ensemble_output_ids_queue.put(next_tokens_id)
            return ensemble_probs.to(scores).detach()

        else:
            next_tokens_id = torch.argmax(scores, dim=-1)
            self.ensemble_output_ids_queue.put(next_tokens_id)
            
            if debug:
                if torch.argmax(scores).item() != self.main_tokenizer.eos_token_id:     # avoid inf value for eos
                    json_object[f'main_tokens'] = (safe_convert_ids_to_tokens(self.main_tokenizer, next_tokens_id), '{:.2f}'.format(main_topK_probs[0, next_tokens_id].item()))
                    json_object[f'ensemble_token'] = json_object[f'main_tokens']
                
                with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                    process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')
            
            return scores


class AssistModelLogitsProcessor(LogitsProcessor):
    """Push assist-model logits into a queue for the main processor."""
    def __init__(self, assist_logits_queue, assist_result_file_path):
        self.assist_logits_queue = assist_logits_queue
        self.assist_result_file_path = assist_result_file_path
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.assist_logits_queue.put(scores)
        return scores
