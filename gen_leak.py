import sys, torch
sys.path.insert(0, "src")
from user_greenlist import make_user_key, green_key, UserGreenListProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
path, uid, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="auto")
V = model.get_output_embeddings().weight.shape[0]
enc = tok("[INST] Explain how diet can affect heart disease. [/INST]", return_tensors="pt").to(model.device)
proc = UserGreenListProcessor(green_key(make_user_key(uid)), V)
with torch.no_grad():
    o = model.generate(**enc, max_new_tokens=300, do_sample=True, temperature=0.7, logits_processor=LogitsProcessorList([proc]))
open(out, "w").write(tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
