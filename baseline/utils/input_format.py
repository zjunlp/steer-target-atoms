
from typing import List

BEGIN="<|begin_of_text|>"
U_END="<|eot_id|>\n"
B_SYS, E_SYS = "<|start_header_id|>", "<|end_header_id|>\n\n"


def llama3_chat_input_format(
    user_input: str,
    model_output: str = "",
    mode: str = "",
) -> List[int]:
    input_content = ""
    if mode == 'personality':
        system_prompt = ""
    elif mode == "safety":
        system_prompt = ""
    else:
        raise NotImplementedError
    if system_prompt != "":
        system_content = B_SYS + "system" + E_SYS + system_prompt + U_END
    else:
        system_content = ""
    input_content = f"{system_content}{B_SYS}user{E_SYS}{user_input}{U_END}{B_SYS}assistant{E_SYS}"
    if model_output != "":
        input_content += f" {model_output.strip()}"
    return input_content

def llama3_chat_input_format_train(
    user_input: str,
    model_output: str = "",
    mode: str = "",
) -> List[int]:
    input_content = ""
    if mode == 'personality':
        system_prompt = """
                        You should think and behave like a person.
                        You can express your thoughts and opinions freely.
                        Just be yourself to answer the following question about your persona.
                        Please answer the question directly in a few sentences.
                        """
    elif mode == "safety":
        system_prompt = ""
    else:
        raise NotImplementedError
    if system_prompt != "":
        system_content = B_SYS + "system" + E_SYS + system_prompt + U_END
    else:
        system_content = ""
    input_content = f"{system_content}{B_SYS}user{E_SYS}{user_input}{U_END}{B_SYS}assistant{E_SYS}"
    if model_output != "":
        input_content += f" {model_output.strip()}"
    return input_content
