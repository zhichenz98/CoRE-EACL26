from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def load_model(model_path, device):
    # if 'Qwen' in model_path:
    #     model = Qwen3ForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype="auto", trust_remote_code=True)
    # else:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype="auto", trust_remote_code=True)
    model = model.eval()
    tokenizer = load_tokenizer(model_path)
    streamer = TextStreamer(tokenizer)
    return model, tokenizer, streamer

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', truncation_side='left',trust_remote_code=True)

    return tokenizer