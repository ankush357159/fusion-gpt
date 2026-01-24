import torch
from transformers import GPT2LMHeadModel


def load_gpt2_weights_into_picogpt(model):
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_sd = hf_model.state_dict()
    sd = model.state_dict()

    mapped = {}

    # Token embeddings
    mapped["embeddings.token_emb.weight"] = hf_sd["transformer.wte.weight"]

    # Positional embeddings (only if not using RoPE)
    if not model.config.use_rope:
        mapped["embeddings.pos_emb.weight"] = hf_sd["transformer.wpe.weight"]

    # Final layer norm
    mapped["ln_f.weight"] = hf_sd["transformer.ln_f.weight"]
    mapped["ln_f.bias"] = hf_sd["transformer.ln_f.bias"]

    # Transformer blocks
    for i in range(model.config.num_layers):
        prefix = f"blocks.{i}."
        hf_prefix = f"transformer.h.{i}."

        mapped[prefix + "ln1.weight"] = hf_sd[hf_prefix + "ln_1.weight"]
        mapped[prefix + "ln1.bias"] = hf_sd[hf_prefix + "ln_1.bias"]
        mapped[prefix + "ln2.weight"] = hf_sd[hf_prefix + "ln_2.weight"]
        mapped[prefix + "ln2.bias"] = hf_sd[hf_prefix + "ln_2.bias"]

        mapped[prefix + "attn.qkv.weight"] = hf_sd[hf_prefix + "attn.c_attn.weight"]
        mapped[prefix + "attn.qkv.bias"] = hf_sd[hf_prefix + "attn.c_attn.bias"]
        mapped[prefix + "attn.proj.weight"] = hf_sd[hf_prefix + "attn.c_proj.weight"]
        mapped[prefix + "attn.proj.bias"] = hf_sd[hf_prefix + "attn.c_proj.bias"]

        mapped[prefix + "ff.net.0.weight"] = hf_sd[hf_prefix + "mlp.c_fc.weight"]
        mapped[prefix + "ff.net.0.bias"] = hf_sd[hf_prefix + "mlp.c_fc.bias"]
        mapped[prefix + "ff.net.2.weight"] = hf_sd[hf_prefix + "mlp.c_proj.weight"]
        mapped[prefix + "ff.net.2.bias"] = hf_sd[hf_prefix + "mlp.c_proj.bias"]

    model.load_state_dict(mapped, strict=False)
    print("GPT-2 weights loaded into PicoGPT (compatible layers)")
