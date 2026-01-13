import transformers
import torch
import os
from tqdm import tqdm
import numpy as np
from .safe_convert_ids_to_tokens import safe_convert_ids_to_tokens
from .calculate_token_map import calculate_token_map_mined, calculate_token_map_unite, calculate_token_map_gac, calculate_token_map_eva


def load_token_map(
    main_model_name,
    assist_model_name,
    main_vocab_size, 
    assist_vocab_size,
    main_model_tokenizer,
    assist_model_tokenizer,
    main_model,
    assist_model,
    device, 
    method='our'
    ):
    """load token map function, this is the main function to call for generating and loading token map

    Args:
        main_model_name (_type_): _description_
        assist_model_name (_type_): _description_
        main_vocab_size (_type_): _description_
        assist_vocab_size (_type_): _description_
        main_model_tokenizer (_type_): _description_
        assist_model_tokenizer (_type_): _description_
        device (_type_): _description_
        method (str, optional): _description_. Defaults to 'our'.

    Returns:
        _type_: _description_
    """
    file_path1 = "./token_map/" +  "{}-{}-{}.pth".format(main_model_name, assist_model_name,method)
    file_path2 = "./token_map/" +  "{}-{}-{}.pth".format(assist_model_name, main_model_name, method)
    token_list_path1 = "./token_map/" +  "{}-{}-{}.txt".format(main_model_name, assist_model_name, method)
    token_list_path2 = "./token_map/" +  "{}-{}-{}.txt".format(assist_model_name, main_model_name, method)
    if method == 'our':
        token_map1 = calculate_token_map_our(main_model_name, assist_model_name, main_vocab_size, assist_vocab_size, main_model_tokenizer, assist_model_tokenizer, file_path1, token_list_path1)
        token_map2 = calculate_token_map_our(assist_model_name, main_model_name, assist_vocab_size, main_vocab_size, assist_model_tokenizer, main_model_tokenizer, file_path2, token_list_path2)
    elif method == 'unite':
        token_map1 = calculate_token_map_unite(main_model_name, assist_model_name, main_vocab_size, assist_vocab_size, main_model_tokenizer, assist_model_tokenizer, file_path1, token_list_path1)
        token_map2 = calculate_token_map_unite(assist_model_name, main_model_name, assist_vocab_size, main_vocab_size, assist_model_tokenizer, main_model_tokenizer, file_path2, token_list_path2)
    elif method == 'mined':
        token_map1 = calculate_token_map_mined(main_model_name, assist_model_name, main_vocab_size, assist_vocab_size, main_model_tokenizer, assist_model_tokenizer, file_path1, token_list_path1)
        token_map2 = calculate_token_map_mined(assist_model_name, main_model_name, assist_vocab_size, main_vocab_size, assist_model_tokenizer, main_model_tokenizer, file_path2, token_list_path2)
    elif method == 'gac':
        token_map1, token_map2 = calculate_token_map_gac(main_model_name, assist_model_name, main_vocab_size, assist_vocab_size, main_model_tokenizer, assist_model_tokenizer, file_path1, file_path2, token_list_path1, token_list_path2)
    elif method == 'eva':
        token_map1 = calculate_token_map_eva(main_model_name, assist_model_name, main_vocab_size, assist_vocab_size, main_model_tokenizer, assist_model_tokenizer, main_model, assist_model, file_path1, token_list_path1)
        token_map2 = calculate_token_map_eva(assist_model_name, main_model_name, assist_vocab_size, main_vocab_size, assist_model_tokenizer, main_model_tokenizer, assist_model, main_model, file_path2, token_list_path2)
    return (token_map1.to(device), token_map2.to(device)) # shape=(main, assist), (assist, main), (main, assist), (assist, main)

def find_common_token(main_model_tokenizer, assist_model_tokenizer):
    main_vocab = main_model_tokenizer.get_vocab()
    main_vocab = {key.replace("Ġ", "▁"): value for key, value in main_vocab.items()}
    assist_vocab = assist_model_tokenizer.get_vocab()
    assist_vocab = {key.replace("Ġ", "▁"): value for key, value in assist_vocab.items()} # unify space format
    commmon_token_mapping = []
    for _, (token, token_id) in enumerate(main_vocab.items()):
        if token in assist_vocab.keys():
            commmon_token_mapping.append([token_id, assist_vocab[token]])
    return torch.tensor(commmon_token_mapping).transpose(0,1) # shapen=(2,n)



def calculate_token_map_our(
    main_model_name,
    assist_model_name,
    main_vocab_size,
    assist_vocab_size, 
    main_model_tokenizer, 
    assist_model_tokenizer, 
    file_path, 
    token_list_path
    ):
    """Our defined token alignment function

    Args:
        main_model_name (_type_): _description_
        assist_model_name (_type_): _description_
        main_vocab_size (_type_): _description_
        assist_vocab_size (_type_): _description_
        main_model_tokenizer (_type_): _description_
        assist_model_tokenizer (_type_): _description_
        file_path (_type_): _description_
        token_list_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    if os.path.exists(file_path):
        return torch.load(file_path)
    common_token_mapping = find_common_token(main_model_tokenizer, assist_model_tokenizer)
    # token_map = torch.sparse_coo_tensor(common_token_mapping, torch.ones(common_token_mapping.size(1)), (main_vocab_size, assist_vocab_size))   # store common mapping
    main_vocab = main_model_tokenizer.get_vocab()
    token_map = torch.zeros((main_vocab_size, assist_vocab_size))
    token_map[common_token_mapping[0], common_token_mapping[1]] = 1
    main_assist_tokens=[]
    num_match, num_partial_match, num_unmatch = 0, 0, 0

    for main_token_id in tqdm(range(main_vocab_size)):
        main_token = main_model_tokenizer.decode([main_token_id])
        if main_token_id in common_token_mapping[0]:  # already matched
            main_assist_tokens.append([main_token, main_token])
            num_match += 1
            continue
        if main_token is None:
            num_unmatch += 1
            continue
        assist_token_id = assist_model_tokenizer.encode(main_token, add_special_tokens=False)
        if len(assist_token_id) == 0:
            num_unmatch += 1
            continue
        elif len(assist_token_id) == 1:
            num_match += 1
        else:
            num_partial_match += 1
        assist_token_id = assist_token_id[0]

        ## post-processing, only add to token map if assist_token is the prefix of main_token
        assist_token = assist_model_tokenizer.decode(assist_token_id)
        if len(assist_token) != 0 and main_token.lower().startswith(assist_token.lower()):
            token_map[main_token_id, assist_token_id] = 1
            main_assist_tokens.append([main_token, assist_token])
    
    print("match: {},{}\n paritially match: {},{}\n unmatch: {},{}".format(num_match, num_match*1.0/main_vocab_size, num_partial_match, num_partial_match*1.0/main_vocab_size, num_unmatch, num_unmatch*1.0/main_vocab_size))
    np.savetxt(token_list_path, np.array(main_assist_tokens), fmt="%s")
    # token_map = token_map / (token_map.sum(dim=0, keepdim=True) + 1e-10)
    token_map = token_map.to_sparse()
    torch.save(token_map, file_path)
    return token_map





if __name__ == "__main__":
    from model_load import load_model
    main_model_name = "InternLM7b"
    assist_model_name = "OpenChat"
    main_model_path = "internlm/internlm2_5-7b-chat"
    assist_model_path = "openchat/openchat-3.5-0106"
    align_method='gac'
    main_model, main_model_tokenizer, _ = load_model(main_model_path, "auto")
    assist_model, assist_model_tokenizer, _ = load_model(assist_model_path, "auto")
    token_map = load_token_map(main_model_name, assist_model_name, main_model.config.vocab_size, assist_model.config.vocab_size, main_model_tokenizer, assist_model_tokenizer, device="cuda:3", method=align_method)
    