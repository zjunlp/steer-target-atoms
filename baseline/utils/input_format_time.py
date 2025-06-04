
from typing import List

BEGIN="<|begin_of_text|>"
U_END="<|eot_id|>\n"
B_SYS, E_SYS = "<|start_header_id|>", "<|end_header_id|>\n\n"
Time_log = "\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"


def llama3_chat_input_time_format(
    user_input: str,
    model_output: str = "",
    system_prompt: str = "",
) -> List[int]:
    input_content = ""
    system_content = B_SYS + "system" + E_SYS + Time_log + system_prompt + U_END
    input_content = f"{system_content}{B_SYS}user{E_SYS}{user_input}{U_END}{B_SYS}assistant{E_SYS}"
    if model_output != "":
        input_content += f" {model_output.strip()}"
    return input_content
