import gradio as gr
import transformers
import torch
from threading import Thread
from accelerate import disk_offload


model_id = "stabilityai/StableBeluga-7B"

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()  # 'cpu'

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map=device,
    offload_folder="offload/"
)
disk_offload(model=model, offload_dir="offload")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)

DESCRIPTION = """
# StableBeluga-7B
This is a test for stableBeluga-7B as a chat interface.
"""

SYS_PRT_EXPLAIN = """
# System Prompt 
Guides you.
"""

prompts = [
    "You are a helpful Ai."
]

def prompt_build(system_prompt, user_inp, hist):
    prompt = f"""### System:\n{system_prompt}\n\n"""
    for pair in hist:
        prompt += f"""### User:\n{pair[0]}\n\n### Assistant:\n{pair[1]}\n\n"""
    prompt += f"""### User:\n{user_inp}\n\n### Assistant:"""
    return prompt

def chat(user_input, history, system_prompt):
    prompt = prompt_build(system_prompt, user_input, history)
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    streamer = transformers.TextIteratorStreamer(
        tokenizer,
        timeout=10.,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_length=4096,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        top_k=50
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    model_output = ""
    for new_text in streamer:
        model_output += new_text
        yield model_output
    return model_output


# GRADIO INTERFACE
with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    gr.Markdown(SYS_PRT_EXPLAIN)
    dropdown = gr.Dropdown(
        choices=prompts,
        label="Select a prompt",
        value="You are cool.",
        allow_custom_value=True
    )
    chatbot = gr.ChatInterface(fn=chat, additional_inputs=[dropdown])


# LAUNCH
demo.queue(api_open=False).launch(server_name="0.0.0.0", show_api=False)