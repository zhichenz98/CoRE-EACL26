import json

def demon_prompt_generate(demon_file_path, demon_parameter):
    with open(demon_file_path, 'r', encoding='utf-8') as demon_file:
        demon_list = demon_file.readlines()
    demon_instruction = ''
    demon_key_list = demon_parameter['key']
    demon_template = demon_parameter['template']
    for demon in demon_list:
        demon = json.loads(demon)
        demon_value_list = [demon[key] for key in demon_key_list]
        demon_instruction += str(demon_template).format(*demon_value_list)
    return demon_instruction, len(demon_list)

def task_instruction_generate(jsonobj, instruction_parameter):
    key_list = instruction_parameter['key']
    template = instruction_parameter['template']
    value_list = [jsonobj[key] for key in key_list]
    instruction = str(template).format(*value_list)
    return instruction

    

# def task_instruction_generate(template, question):
#     key_list = template["instruction_parameter"]["key"]
#     value_list = [question[key] for key in key_list]
#     task_instruction = str(template["instruction_parameter"]["template"]).format(*value_list)
#     final_input_prompt = template["instruction"] + task_instruction
#     main_model_input = template["main_model_system_template"].format(final_input_prompt)    # <s> + Demons + Q: xxx, A:
#     assist_model_input = template["assist_model_system_template"].format(final_input_prompt)    # <s> + Demons + Q: xxx, A:
#     return {"main_model_input": main_model_input, "assist_model_input": assist_model_input}
