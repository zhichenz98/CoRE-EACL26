import os
from typing import Dict, Tuple, List
from tqdm import tqdm

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from .safe_convert_ids_to_tokens import safe_convert_ids_to_tokens

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

# def calculate_token_map_unite(
#     main_model_name,
#     assist_model_name,
#     main_vocab_size,
#     assist_vocab_size,
#     main_model_tokenizer,
#     assist_model_tokenizer,
#     file_path,
#     token_list_path
#     ):
#     # Ensure the output directory exists.
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     # If an alignment has been saved previously, load and return it.
#     if os.path.exists(file_path):
#         return torch.load(file_path)
#     common_token_mapping = find_common_token(main_model_tokenizer, assist_model_tokenizer)
#     # token_map = torch.sparse_coo_tensor(common_token_mapping, torch.ones(common_token_mapping.size(1)), (main_vocab_size, assist_vocab_size))   # store common mapping
#     main_vocab = main_model_tokenizer.get_vocab()
#     token_map = torch.zeros((main_vocab_size, assist_vocab_size))
#     token_map[common_token_mapping[0], common_token_mapping[1]] = 1
#     main_assist_tokens=[]
#     num_match, num_partial_match, num_unmatch = 0, 0, 0

#     for main_token_id in tqdm(range(main_vocab_size)):
#         main_token = main_model_tokenizer.decode([main_token_id])
#         if main_token_id in common_token_mapping[0]:  # already matched
#             main_assist_tokens.append([main_token, main_token])
#             num_match += 1
#             continue
#         if main_token is None:
#             num_unmatch += 1
#             continue
#         assist_token_id = assist_model_tokenizer.encode(main_token, add_special_tokens=False)
#         if len(assist_token_id) == 0:
#             num_unmatch += 1
#             continue
#         elif len(assist_token_id) == 1:
#             num_match += 1
#         else:
#             num_partial_match += 1
#         assist_token_id = assist_token_id[0]

#         ## post-processing, only add to token map if assist_token is the prefix of main_token
#         assist_token = assist_model_tokenizer.decode(assist_token_id)
#         if len(assist_token) != 0 and main_token.lower().startswith(assist_token.lower()):
#             token_map[main_token_id, assist_token_id] = 1
#             main_assist_tokens.append([main_token, assist_token])
    
#     print("match: {},{}\n paritially match: {},{}\n unmatch: {},{}".format(num_match, num_match*1.0/main_vocab_size, num_partial_match, num_partial_match*1.0/main_vocab_size, num_unmatch, num_unmatch*1.0/main_vocab_size))
#     np.savetxt(token_list_path, np.array(main_assist_tokens), fmt="%s")
#     # token_map = token_map / (token_map.sum(dim=0, keepdim=True) + 1e-10)
#     token_map = token_map.to_sparse()
#     torch.save(token_map, file_path)
#     return token_map



# def calculate_token_map_unite(
#     main_model_name,
#     assist_model_name,
#     main_vocab_size,
#     assist_vocab_size,
#     main_model_tokenizer,
#     assist_model_tokenizer,
#     file_path,
#     token_list_path
#     ):
#     """self-implemented token alignment via UniTE method.

#     Args:
#         main_model_name (_type_): _description_
#         assist_model_name (_type_): _description_
#         main_vocab_size (_type_): _description_
#         assist_vocab_size (_type_): _description_
#         main_model_tokenizer (_type_): _description_
#         assist_model_tokenizer (_type_): _description_
#         file_path (_type_): _description_
#         token_list_path (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     if os.path.exists(file_path):
#         return torch.load(file_path)
#     main_vocab = main_model_tokenizer.get_vocab()
#     token_map = torch.zeros((main_vocab_size, assist_vocab_size))
#     main_assist_tokens=[]
#     num_match, num_partial_match, num_unmatch = 0, 0, 0
    
#     main_vocab = main_model_tokenizer.get_vocab()
#     for _, (_, main_token_id) in tqdm(enumerate(main_vocab.items())):
#         main_token = safe_convert_ids_to_tokens(main_model_tokenizer, main_token_id, skip_special_tokens=True)
        
#         # normalize
#         # token = main_token.replace('Ġ','_').replace('<0x0A>','\n').replace('Ċ','\n')
#         # if 'llama' in assist_model_name.lower() or 'qwen' in assist_model_name.lower():
#         #     token = token.replace('▁','Ġ')
        
#         if main_model_name in ['Llama2', 'Llama3', 'OpenChat', 'Mistral'] and assist_model_name in ['Llama2', 'Llama3', 'OpenChat', 'Mistral']:
#             token = main_token.replace('▁','Ġ').replace('Ċ','/n')
#         else:
#             token = main_token.replace('▁','Ġ').replace('<0x0A>','/n').replace('Ċ','/n')
#         if 'llama' in assist_model_name.lower() or 'mistral' in assist_model_name.lower() or 'openchat' in assist_model_name.lower() or 'internlm' in assist_model_name.lower() or 'qwen' in assist_model_name.lower():
#             token = token.replace('Ġ','▁')
            
#         if token != '':
#             subtoken_id = assist_model_tokenizer.convert_tokens_to_ids(token)
#             if subtoken_id != 0 and subtoken_id != None: #Mistral and Llama2 oov id 0
#                 main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, subtoken_id)])
#                 token_map[main_token_id, subtoken_id] = 1
#                 num_match += 1
#             else:
#                 subtokens = assist_model_tokenizer.tokenize(token)
#                 for token_id in assist_model_tokenizer.convert_tokens_to_ids(subtokens):
#                     if 'llama3' in assist_model_name.lower():
#                         if token_id != 220:
#                             main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
#                             token_map[main_token_id, token_id] = 1
#                             num_partial_match += 1
#                             break
#                     if 'llama2' in assist_model_name.lower():
#                         if token_id != 29871:
#                             main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
#                             token_map[main_token_id, token_id] = 1
#                             num_partial_match += 1
#                             break
#                     if 'mistral' in assist_model_name.lower():
#                         if token_id != 29473:
#                             main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
#                             token_map[main_token_id, token_id] = 1
#                             num_partial_match += 1
#                             break
#                     if 'deepseek' in assist_model_name.lower():
#                         if token_id != 207:
#                             main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
#                             token_map[main_token_id, token_id] = 1
#                             num_partial_match += 1
#                             break
#                     if 'openchat' in assist_model_name.lower():
#                         if token_id != 28705:
#                             main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
#                             token_map[main_token_id, token_id] = 1
#                             num_partial_match += 1
#                             break
#                     if 'qwen' in assist_model_name.lower():
#                         if token_id != 220:
#                             main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
#                             token_map[main_token_id, token_id] = 1
#                             num_partial_match += 1
#                             break
#                     else:
#                         main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
#                         token_map[main_token_id, token_id] = 1
#                         num_partial_match += 1
#                         break
#         else:
#             if 'llama3' in assist_model_name.lower() or 'qwen2' in assist_model_name.lower():
#                 main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, 220)])
#                 token_map[main_token_id, 220] = 1
#             if 'llama2' in assist_model_name.lower():
#                 main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, 29871)])
#                 token_map[main_token_id, 29871] = 1
#             if 'mistral' in assist_model_name.lower():
#                 main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, 29473)])
#                 token_map[main_token_id, 29473] = 1
#             if 'deepseek' in assist_model_name.lower():
#                 main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, 207)])
#                 token_map[main_token_id, 207] = 1
    
#     num_unmatch = main_vocab_size - num_match - num_partial_match
#     print("unite_align\nmatch: {},{}\n paritially match: {},{}\n unmatch: {},{}\n".format(num_match, num_match*1.0/main_vocab_size, num_partial_match, num_partial_match*1.0/main_vocab_size, num_unmatch, num_unmatch*1.0/main_vocab_size))
#     np.savetxt(token_list_path, np.array(main_assist_tokens), fmt="%s")
#     token_map = token_map.to_sparse()
#     torch.save(token_map, file_path)
#     return token_map

def calculate_token_map_unite(main_model_name, assist_model_name, main_vocab_size, assist_vocab_size, main_model_tokenizer, assist_model_tokenizer, file_path, token_list_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # If an alignment has been saved previously, load and return it.
    if os.path.exists(file_path):
        return torch.load(file_path)
   
    main_vocab = main_model_tokenizer.get_vocab()
    token_map = torch.zeros((main_vocab_size, assist_vocab_size))
    main_assist_tokens=[]
    num_match, num_partial_match, num_unmatch = 0, 0, 0
    
    main_vocab = main_model_tokenizer.get_vocab()
    for _, (_, main_token_id) in tqdm(enumerate(main_vocab.items())):
        main_token = safe_convert_ids_to_tokens(main_model_tokenizer, main_token_id, skip_special_tokens=True)
        if main_model_name in ['Llama2', 'Llama3', 'OpenChat', 'Mistral'] and assist_model_name in ['Llama2', 'Llama3', 'OpenChat', 'Mistral']:
            token = main_token.replace('▁','Ġ').replace('Ċ','\n')
        else:
            token = main_token.replace('▁','Ġ').replace('<0x0A>','\n').replace('Ċ','\n')
        if 'llama2' in assist_model_name.lower() or 'mistral' in assist_model_name.lower() or 'openchat' in assist_model_name.lower() or 'internlm' in assist_model_name.lower() or 'qwen' in assist_model_name.lower():
            token = token.replace('Ġ','▁')
        if token != '':
            subtoken_id = assist_model_tokenizer.convert_tokens_to_ids(token)
            if subtoken_id != 0 and subtoken_id != None: #Mistral and Llama2 oov id 0
                main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, subtoken_id)])
                token_map[main_token_id, subtoken_id] = 1
                num_match += 1
            else:
                subtokens = assist_model_tokenizer.tokenize(token)
                for token_id in assist_model_tokenizer.convert_tokens_to_ids(subtokens):
                    if 'llama3' in assist_model_name.lower():
                        if token_id != 220:
                            main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
                            token_map[main_token_id, token_id] = 1
                            num_partial_match += 1
                            break
                    if 'llama2' in assist_model_name.lower():
                        if token_id != 29871:
                            main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
                            token_map[main_token_id, token_id] = 1
                            num_partial_match += 1
                            break
                    if 'mistral' in assist_model_name.lower():
                        if token_id != 29473:
                            main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
                            token_map[main_token_id, token_id] = 1
                            num_partial_match += 1
                            break
                    if 'deepseek' in assist_model_name.lower():
                        if token_id != 207:
                            main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
                            token_map[main_token_id, token_id] = 1
                            num_partial_match += 1
                            break
                    if 'openchat' in assist_model_name.lower():
                        if token_id != 28705:
                            main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
                            token_map[main_token_id, token_id] = 1
                            num_partial_match += 1
                            break
                    else:
                        main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, token_id)])
                        token_map[main_token_id, token_id] = 1
                        num_partial_match += 1
                        break
        else:
            if 'llama3' in assist_model_name.lower() or 'qwen2' in assist_model_name.lower():
                main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, 220)])
                token_map[main_token_id, 220] = 1
            if 'llama2' in assist_model_name.lower():
                main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, 29871)])
                token_map[main_token_id, 29871] = 1
            if 'mistral' in assist_model_name.lower():
                main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, 29473)])
                token_map[main_token_id, 29473] = 1
            if 'deepseek' in assist_model_name.lower():
                main_assist_tokens.append([main_token, safe_convert_ids_to_tokens(assist_model_tokenizer, 207)])
                token_map[main_token_id, 207] = 1
    
    num_unmatch = main_vocab_size - num_match - num_partial_match
    print("unite_align\nmatch: {},{}\n paritially match: {},{}\n unmatch: {},{}\n".format(num_match, num_match*1.0/main_vocab_size, num_partial_match, num_partial_match*1.0/main_vocab_size, num_unmatch, num_unmatch*1.0/main_vocab_size))
    np.savetxt(token_list_path, np.array(main_assist_tokens), fmt="%s")
    token_map = token_map.to_sparse()
    torch.save(token_map, file_path)
    return token_map

def _levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    This function implements a standard dynamic programming algorithm
    with O(len(a) * len(b)) time and O(len(b)) auxiliary space.  It
    returns the minimum number of single‑character insertions,
    deletions or substitutions required to transform ``a`` into ``b``.

    Parameters
    ----------
    a : str
        The source string.
    b : str
        The target string.

    Returns
    -------
    int
        The Levenshtein distance between ``a`` and ``b``.
    """
    # Short‑circuit trivial cases.
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure ``b`` is the longer string to minimise memory footprint.
    if len(a) > len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        # ``previous_row[j]`` corresponds to distance(a[:i−1], b[:j])
        # ``current_row[j]`` corresponds to distance(a[:i], b[:j])
        for j, cb in enumerate(b, start=1):
            insertions = previous_row[j] + 1      # insertion in ``a``
            deletions = current_row[j - 1] + 1    # deletion from ``a``
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _build_id_to_token(tokenizer: PreTrainedTokenizerBase) -> Dict[int, str]:
    """Construct a mapping from token id to token text with unified whitespace.

    HuggingFace tokenizers encode leading whitespace with special
    sentinel characters such as ``Ġ`` or ``▁`` depending on the model.
    To facilitate comparison between different vocabularies, we
    normalise these sentinel characters to a common form (``▁``).  If
    ``get_vocab`` returns duplicate ids for different tokens (which
    should not happen in well‑formed tokenizers), the later ones will
    overwrite earlier ones.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer whose vocabulary is to be inverted.

    Returns
    -------
    dict[int, str]
        A dictionary mapping token ids to their normalised token texts.
    """
    vocab = tokenizer.get_vocab()
    id_to_token: Dict[int, str] = {}
    # for token, token_id in vocab.items():
    print('normalizing token space...')
    for _, (_, token_id) in tqdm(enumerate(vocab.items())):
        token = safe_convert_ids_to_tokens(tokenizer, token_id, skip_special_tokens=True)
        # normalize tokens
        token = token.replace('Ġ','_').replace('<0x0A>','\n').replace('Ċ','\n')

        id_to_token[token_id] = token
    return id_to_token


def calculate_token_map_mined(
    main_model_name: str,
    assist_model_name: str,
    main_vocab_size: int,
    assist_vocab_size: int,
    main_model_tokenizer: PreTrainedTokenizerBase,
    assist_model_tokenizer: PreTrainedTokenizerBase,
    file_path: str,
    token_list_path: str,
) -> torch.Tensor:
    """Align two vocabularies using minimum edit distance with a trie‑based DP.

    For each token id ``i`` in the main vocabulary, this function
    identifies a token id ``j`` in the assist vocabulary whose token
    string minimises the Levenshtein distance to the main token.  A
    sparse matrix ``A`` of shape ``(main_vocab_size, assist_vocab_size)``
    is constructed such that ``A[i, j] = 1`` for the chosen ``j`` and
    all other entries are zero.  The mapping pairs (main token text,
    assist token text) are saved as a plain‑text file for inspection.

    Compared to a naive O(V1·V2) nested loop, this implementation
    leverages a trie and dynamic programming search to prune the
    candidate space when computing edit distances.  The algorithm
    described here is inspired by fuzzy string matching techniques
    using tries.  It dramatically reduces the number of character
    comparisons by exploring only those prefixes whose partial edit
    distances are competitive with the best found so far.

    Parameters
    ----------
    main_model_name : str
        Name of the main model (unused but kept for API parity).
    assist_model_name : str
        Name of the assist model (unused but kept for API parity).
    main_vocab_size : int
        Size of the main model's vocabulary.
    assist_vocab_size : int
        Size of the assist model's vocabulary.
    main_model_tokenizer : PreTrainedTokenizerBase
        Tokenizer for the main model.
    assist_model_tokenizer : PreTrainedTokenizerBase
        Tokenizer for the assist model.
    file_path : str
        Path to save the resulting sparse token map (``.pth`` file).
    token_list_path : str
        Path to save the human‑readable token pairs (``.txt`` file).

    Returns
    -------
    torch.Tensor
        A sparse tensor of shape ``(main_vocab_size, assist_vocab_size)``
        with ``main_vocab_size`` non‑zero values indicating the minimal
        edit distance alignments.
    """
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # If an alignment has been saved previously, load and return it.
    if os.path.exists(file_path):
        return torch.load(file_path)

    # Build id→token mappings for both tokenizers.  Normalising the
    # whitespace sentinel ensures that tokens representing the same
    # string with different encodings (e.g. ``Ġthe`` vs ``▁the``) can
    # match exactly via edit distance 0.
    id_to_main = _build_id_to_token(main_model_tokenizer)
    id_to_assist = _build_id_to_token(assist_model_tokenizer)

    # Build a trie from the assist vocabulary to accelerate edit distance
    # lookup.  Each node stores a mapping from character to child
    # nodes, a flag indicating if a token ends here, and the token id.
    class _TrieNode:
        __slots__ = ("children", "token_id", "is_word")

        def __init__(self):
            self.children: Dict[str, "_TrieNode"] = {}
            self.token_id: int = -1
            self.is_word: bool = False

    root = _TrieNode()
    # Normalise and lowercase tokens when inserting into the trie.
    for assist_id in range(assist_vocab_size):
        token = id_to_assist.get(assist_id, "")
        if not token:
            continue
        key = token.lower()
        node = root
        for ch in key:
            node = node.children.setdefault(ch, _TrieNode())
        # Only store the first token id for this string to avoid
        # overwriting earlier entries.  If the vocabulary contains
        # duplicate normalised tokens, subsequent duplicates are ignored.
        if not node.is_word:
            node.is_word = True
            node.token_id = assist_id

    # Prepare data structures to build the sparse tensor.  We store
    # indices and values separately before constructing the tensor at
    # the end.  Only one assist index will be selected per main token.
    indices: List[List[int]] = [[], []]  # shape (2, num_nonzero)
    values: List[float] = []
    token_pairs: List[Tuple[str, str]] = []

    num_exact = 0
    num_non_exact = 0
    num_missing = 0

    # Helper function to recursively search the trie for the best match.
    def _search_trie(node: _TrieNode, prev_row: List[int], pattern: str, best: List[int]):
        """Recursive helper for approximate search on the trie.

        Parameters
        ----------
        node : _TrieNode
            The current trie node being explored.
        prev_row : List[int]
            The DP row corresponding to the edit distances between the
            pattern and the prefix represented by the current node.
        pattern : str
            The target string whose nearest neighbour is sought.
        best : List[int]
            A two element list ``[best_distance, best_id]`` storing the
            current best edit distance and the associated assist token id.
        """
        pattern_len = len(pattern)
        # If this node corresponds to a complete word, update the best
        # match if the edit distance is improved.
        if node.is_word:
            dist = prev_row[pattern_len]
            if dist < best[0]:
                best[0] = dist
                best[1] = node.token_id
        # Explore children.  If the smallest value in the current row
        # exceeds or equals the best distance found so far, prune the
        # branch.  This implements a branch‑and‑bound strategy.
        if min(prev_row) >= best[0]:
            return
        for ch, child in node.children.items():
            # Compute current row for this child by extending the prefix with
            # character ``ch``.  ``prev_row`` is of length pattern_len+1.
            curr_row = [prev_row[0] + 1]
            for j in range(1, pattern_len + 1):
                insert_cost = curr_row[j - 1] + 1
                delete_cost = prev_row[j] + 1
                replace_cost = prev_row[j - 1] + (pattern[j - 1] != ch)
                curr_row.append(min(insert_cost, delete_cost, replace_cost))
            # Only recurse if this path could still yield a better match.
            if min(curr_row) <= best[0]:
                _search_trie(child, curr_row, pattern, best)

    # Iterate over all possible main token ids.  Some ids might not
    # correspond to valid entries in ``id_to_main`` (e.g. special
    # tokens); those are skipped to avoid creating spurious alignments.
    for main_id in tqdm(range(main_vocab_size)):
        main_token = id_to_main.get(main_id, "")
        if not main_token:
            # Skip empty or unknown tokens.
            num_missing += 1
            continue
        pattern = main_token.lower()
        # Initialise DP row for empty prefix: distance to each prefix of the pattern.
        initial_row = list(range(len(pattern) + 1))
        # ``best`` holds the current best edit distance and the corresponding assist id.
        best = [float("inf"), None]
        # Kick off the recursive search.  This call will update ``best``.
        _search_trie(root, initial_row, pattern, best)
        best_distance, best_assist_id = best[0], best[1]
        if best_assist_id is None:
            num_missing += 1
            continue
        # Record the mapping.
        indices[0].append(main_id)
        indices[1].append(best_assist_id)
        values.append(1.0)
        token_pairs.append((main_token, id_to_assist[best_assist_id]))
        if best_distance == 0:
            num_exact += 1
        else:
            num_non_exact += 1

    total_aligned = num_exact + num_non_exact
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    values_tensor = torch.tensor(values, dtype=torch.float)
    token_map = torch.sparse_coo_tensor(
        indices=indices_tensor,
        values=values_tensor,
        size=(main_vocab_size, assist_vocab_size)
    )

    torch.save(token_map, file_path)
    np.savetxt(token_list_path, np.array(token_pairs), fmt="%s\t%s")

    print(
        f"minED_align_DP\n"
        f"exact matches: {num_exact}, fraction: {num_exact / max(total_aligned, 1):.4f}\n"
        f"non‑exact matches: {num_non_exact}, fraction: {num_non_exact / max(total_aligned, 1):.4f}\n"
        f"skipped: {num_missing}, fraction: {num_missing / float(main_vocab_size):.4f}\n"
    )

    return token_map




def calculate_token_map_gac(
    main_model_name: str,
    assist_model_name: str,
    main_vocab_size: int,
    assist_vocab_size: int,
    main_model_tokenizer: PreTrainedTokenizerBase,
    assist_model_tokenizer: PreTrainedTokenizerBase,
    main_file_path: str,
    assist_file_path: str,
    main_token_list_path: str,
    assist_token_list_path: str,
) -> torch.Tensor:
    """Construct a one‑hot union mapping for GaC ensembling.

    This function realises the token alignment described in the GAC paper.
    Given two tokenizers with vocabularies
    ``V1`` and ``V2``, it forms the union vocabulary ``V = V1 ∪ V2`` and
    builds a sparse matrix ``A`` of shape ``(main_vocab_size, |V|)`` such
    that each row contains a single 1 at the column corresponding to the
    union index of the main token.  Because ``V`` contains all tokens
    from both vocabularies, every token from the main vocabulary has a
    well‑defined position in the union.  This matrix can be used to map
    probability vectors from the main model into the union space via
    ``p · A``.  When called twice with swapped arguments, two such
    matrices can be obtained for both models, allowing their output
    probabilities to be ensembled by averaging in the union space.

    Parameters
    ----------
    main_model_name : str
        Name of the main model (unused but kept for API parity).
    assist_model_name : str
        Name of the assist model (unused but kept for API parity).
    main_vocab_size : int
        Size of the main model's vocabulary.
    assist_vocab_size : int
        Size of the assist model's vocabulary.
    main_model_tokenizer : PreTrainedTokenizerBase
        Tokenizer for the main model.
    assist_model_tokenizer : PreTrainedTokenizerBase
        Tokenizer for the assist model.
    file_path : str
        Path to save the resulting sparse token map (``.pth`` file).
    token_list_path : str
        Path to save the list of union tokens (``.txt`` file).

    Returns
    -------
    torch.Tensor
        A sparse tensor ``A`` of shape ``(main_vocab_size, |V|)`` where
        ``A[i, j] = 1`` if the ``i``‑th token of the main vocabulary
        corresponds to the ``j``‑th token in the union vocabulary.
    """
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(main_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(assist_file_path), exist_ok=True)
    if os.path.exists(main_file_path) and os.path.exists(assist_file_path):
        return torch.load(main_file_path), torch.load(assist_file_path)
    
    # Build id→token mappings for both tokenizers (normalised).
    id_to_main = _build_id_to_token(main_model_tokenizer)
    id_to_assist = _build_id_to_token(assist_model_tokenizer)
    union_tokens = sorted(set(id_to_main.values()) | set(id_to_assist.values()))
    union_size = len(union_tokens)
    # Map token string → union index.
    union_dict: Dict[str, int] = {tok: idx for idx, tok in enumerate(union_tokens)}
    # Create a sparse mapping for the main vocabulary.  Each row maps
    # the main token id to its union index.  Rows corresponding to
    # unknown tokens remain zero (no entry).
    main_row_indices: List[int] = []
    main_col_indices: List[int] = []
    for main_token_id in tqdm(range(main_vocab_size)):
        main_token = id_to_main.get(main_token_id, "")
        if not main_token:
            continue
        main_union_idx = union_dict[main_token]
        main_row_indices.append(main_token_id)
        main_col_indices.append(main_union_idx)
    main_indices_tensor = torch.tensor([main_row_indices, main_col_indices], dtype=torch.long)
    main_values_tensor = torch.ones(len(main_row_indices), dtype=torch.float)
    main_union_token_map = torch.sparse_coo_tensor(
        indices=main_indices_tensor,
        values=main_values_tensor,
        size=(main_vocab_size, union_size)
    )
    
    assist_row_indices: List[int] = []
    assist_col_indices: List[int] = []
    for assist_token_id in tqdm(range(assist_vocab_size)):
        assist_token = id_to_assist.get(assist_token_id, "")
        if not assist_token:
            continue
        assist_union_idx = union_dict[assist_token]
        assist_row_indices.append(assist_token_id)
        assist_col_indices.append(assist_union_idx)
    assist_indices_tensor = torch.tensor([assist_row_indices, assist_col_indices], dtype=torch.long)
    assist_values_tensor = torch.ones(len(assist_row_indices), dtype=torch.float)
    assist_union_token_map = torch.sparse_coo_tensor(
        indices=assist_indices_tensor,
        values=assist_values_tensor,
        size=(assist_vocab_size, union_size)
    )
    # Persist the mapping and the union tokens for inspection.  We
    # save the union tokens to ``token_list_path`` for future use by
    # downstream components (e.g. reverse mapping back to strings).
    
    main_to_assist_map = main_union_token_map.to_dense() @ assist_union_token_map.to_dense().T
    assist_to_main_map = main_to_assist_map.T.to_sparse()
    main_to_assist_map = main_to_assist_map.to_sparse()
    torch.save(main_to_assist_map, main_file_path)
    torch.save(assist_to_main_map, assist_file_path)
    np.savetxt(main_token_list_path, np.array(union_tokens), fmt="%s")
    np.savetxt(assist_token_list_path, np.array(union_tokens), fmt="%s")
    return main_to_assist_map, assist_to_main_map



def calculate_token_map_eva(
    main_model_name: str,
    assist_model_name: str,
    main_vocab_size: int,
    assist_vocab_size: int,
    main_model_tokenizer: PreTrainedTokenizerBase,
    assist_model_tokenizer: PreTrainedTokenizerBase,
    main_model,
    assist_model,
    file_path: str,
    token_list_path: str,
    top_k: int = 1,
    similarity_threshold: float = None,
) -> torch.Tensor:
    """Align two vocabularies using EVA's embedding projection method.

    This function implements a simplified version of the EVA vocabulary
    projection described in the paper.  It aligns the
    token embeddings of a main model to those of an assist model via an
    orthogonal transformation learned from overlapping tokens.  The
    resulting similarity matrix provides soft or hard mappings between
    the vocabularies.

    Parameters
    ----------
    main_model_name : str
        Name of the main model (unused but kept for API parity).
    assist_model_name : str
        Name of the assist model (unused but kept for API parity).
    main_model : torch.nn.Module
        The main LLM model, from which input embeddings are extracted.
    assist_model : torch.nn.Module
        The assist LLM model.
    main_model_tokenizer : PreTrainedTokenizerBase
        Tokenizer corresponding to the main model.
    assist_model_tokenizer : PreTrainedTokenizerBase
        Tokenizer corresponding to the assist model.
    file_path : str
        Path to save the resulting sparse token map (``.pth`` file).
    token_list_path : str
        Path to save a list of mapping pairs and similarity scores.
    top_k : int, optional
        Number of top similar assist tokens to keep for each main token.
        Defaults to 1, which yields a hard alignment.  Setting ``top_k``
        greater than 1 will produce a sparse matrix with up to ``top_k``
        non‑zero entries per row.
    similarity_threshold : float, optional
        If set, similarity scores below this threshold are discarded.
        This can be used to prune low‑quality mappings as suggested in
        the EVA paper.  If ``None``, no threshold is
        applied.

    Returns
    -------
    torch.Tensor
        A sparse tensor of shape ``(main_vocab_size, assist_vocab_size)``
        containing the alignment scores.  When ``top_k = 1`` and
        ``similarity_threshold`` is ``None``, each row contains a single
        non‑zero entry corresponding to the most similar assist token.
    """
    # Ensure the output directory exists and attempt to load a cached
    # mapping if present.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        return torch.load(file_path)
    # Helper to find overlapping tokens between two vocabularies.  We
    # normalise the tokens by replacing ``Ġ`` with ``▁`` so that
    # identical words can be detected across different tokenisation
    # schemes.
    def _find_common_token_ids(tok1: PreTrainedTokenizerBase, tok2: PreTrainedTokenizerBase) -> Tuple[torch.Tensor, torch.Tensor]:
        vocab1 = tok1.get_vocab()
        vocab2 = tok2.get_vocab()
        # Build maps from normalised token text to id for both vocabs.
        norm1: Dict[str, int] = {}
        for t, i in vocab1.items():
            norm = t.replace("Ġ", "▁")
            if norm not in norm1:
                norm1[norm] = i
        norm2: Dict[str, int] = {}
        for t, i in vocab2.items():
            norm = t.replace("Ġ", "▁")
            if norm not in norm2:
                norm2[norm] = i
        common_tokens = set(norm1.keys()) & set(norm2.keys())
        ids1 = []
        ids2 = []
        for tok in common_tokens:
            ids1.append(norm1[tok])
            ids2.append(norm2[tok])
        # Return as tensors on CPU.
        return torch.tensor(ids1, dtype=torch.long), torch.tensor(ids2, dtype=torch.long)
    # Extract the embedding matrices.  For causal LLMs, the input
    # embeddings reside in the ``lm_head`` or ``embed_tokens`` module.
    # We access them via ``get_input_embeddings`` for generality.
    with torch.no_grad():
        E1: torch.Tensor = main_model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
        E2: torch.Tensor = assist_model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
    # Identify overlapping tokens and extract corresponding embeddings.
    ids1, ids2 = _find_common_token_ids(main_model_tokenizer, assist_model_tokenizer)
    if len(ids1) == 0:
        raise ValueError("No overlapping tokens found between the two vocabularies.")
    X = E1[ids1]  # shape [n_overlap, d]
    Y = E2[ids2]  # shape [n_overlap, d]
    # Centre the embeddings by subtracting the mean.  This helps
    # stabilise the orthogonal Procrustes solution.
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    Xc = X - X_mean
    Yc = Y - Y_mean
    # Compute the cross covariance and obtain the orthogonal matrix via SVD.
    # cross_cov = Xc.T @ Yc ; shape [d, d]
    cross_cov = Xc.t().mm(Yc).to(torch.float32)
    # Compute SVD.  ``torch.linalg.svd`` returns U, S, Vh such that
    # cross_cov = U @ diag(S) @ Vh.  The optimal orthogonal mapping is
    # W = U @ Vh.
    U, _, Vh = torch.linalg.svd(cross_cov, full_matrices=False)
    W = U.mm(Vh)
    # Project the entire main embedding matrix to the assist space.
    E1_proj = (E1 - X_mean).mm(W) + Y_mean  # Optional: add back mean of Y.
    # Normalise embeddings to unit vectors for cosine similarity.  Add
    # a small epsilon to avoid division by zero.
    epsilon = 1e-12
    E1_norm = E1_proj / (E1_proj.norm(dim=1, keepdim=True) + epsilon)
    E2_norm = E2 / (E2.norm(dim=1, keepdim=True) + epsilon)
    # Pre‑compute the transpose of E2_norm for efficient dot products.
    E2_norm_t = E2_norm.t()
    # Prepare lists for sparse indices and values, and record mapping for
    # inspection.  ``top_k`` may yield multiple matches per row.
    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []
    mapping_pairs: List[Tuple[str, str, float]] = []
    # Iterate over each main token to find the top‑k most similar assist
    # tokens.  For efficiency, we compute the similarity vector for the
    # current row by matrix multiplication and then apply ``topk``.
    vocab_size1 = E1_norm.size(0)
    for i in tqdm(range(vocab_size1)):
        sim_row = E1_norm[i].unsqueeze(0).mm(E2_norm_t).squeeze(0)  # shape [vocab2]
        # Apply threshold if specified by zeroing values below the
        # threshold before top‑k.  This reduces candidate matches and
        # enforces quality constraints【187079302551847†L442-L499】.
        if similarity_threshold is not None:
            sim_row = torch.where(sim_row >= similarity_threshold, sim_row, torch.tensor(-float('inf')))
        # Determine how many candidates to retain.  ``top_k`` cannot exceed
        # the vocabulary size of the assist model.
        k = min(top_k, sim_row.numel())
        # Obtain the indices of the top‑k similarities.  ``largest=True``
        # ensures descending order.
        top_vals, top_idxs = torch.topk(sim_row, k=k, largest=True)
        # Keep only finite similarities (skip -inf).  This prevents
        # spurious entries when thresholding.
        for val, j in zip(top_vals.tolist(), top_idxs.tolist()):
            if val == -float('inf'):
                continue
            row_indices.append(i)
            col_indices.append(j)
            values.append(val)
            # Record the mapping pair and similarity score.
            tok_main = main_model_tokenizer.convert_ids_to_tokens([i])[0]
            tok_assist = assist_model_tokenizer.convert_ids_to_tokens([j])[0]
            mapping_pairs.append((tok_main, tok_assist, val))
    # Build the sparse similarity matrix.
    indices_tensor = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values_tensor = torch.tensor(values, dtype=torch.float)
    token_map = torch.sparse_coo_tensor(
        indices=indices_tensor,
        values=values_tensor,
        size=(E1.size(0), E2.size(0))
    )
    # Save the mapping and the mapping pairs for analysis.
    torch.save(token_map, file_path)
    # Save as text: each row contains main token, assist token, similarity.
    with open(token_list_path, 'w', encoding='utf-8') as f:
        for m_tok, a_tok, score in mapping_pairs:
            f.write(f"{m_tok}\t{a_tok}\t{score}\n")
    return token_map