import sys
import torch
torch.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

base_model = 'baffo32/decapoda-research-llama-7B-hf'
model_type = 'alpaca-lora-7b'
load_8bit = False
lora_weights = './TALLRec/alpaca-lora-7B'

tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map={'': 0}
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        base_model, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
    )

tokenizer.padding_side = "left"
# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

########inference#########

instructions=["Given the user's preference and unpreference, identify whether the user will like the target movie by answering \"Yes.\" or \"No.\"."]
inputs=["User Preference: \"Paris, Texas (1984)\", \"Rebel Without a Cause (1955)\", \"Return of the Pink Panther, The (1974)\", \"Ace Ventura: Pet Detective (1994)\", \"Magnificent Seven, The (1954)\", \"Star Trek: The Wrath of Khan (1982)\", \"Cat People (1982)\", \"Orlando (1993)\", \"Dave (1993)\"\nUser Unpreference: \"Kalifornia (1993)\"\nWhether the user will like the target movie \"Perez Family, The (1995)\"?"]
temperature=1
top_p=1.0
top_k=40
num_beams=1
max_new_tokens=128
batch_size=1

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""

prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
generation_config = GenerationConfig(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    num_beams=num_beams
)
with torch.no_grad():
    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
        # batch_size=batch_size,
    )
s = generation_output.sequences
scores = generation_output.scores[0].softmax(dim=-1)
logits = torch.tensor(scores[:,[8241, 3782]], dtype=torch.float32).softmax(dim=-1)
input_ids = inputs["input_ids"].to(device)
L = input_ids.shape[1]
s = generation_output.sequences
output = tokenizer.batch_decode(s, skip_special_tokens=True)
output = [_.split('Response:\n')[-1] for _ in output]

print(output[0])