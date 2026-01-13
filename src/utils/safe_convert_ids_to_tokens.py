def safe_convert_ids_to_tokens(tokenizer, id: int, skip_special_tokens=False):
    token = tokenizer.convert_ids_to_tokens(id, skip_special_tokens=skip_special_tokens)
    return token.decode("utf-8") if isinstance(token, bytes) else token