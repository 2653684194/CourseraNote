import os
import math
from typing import Tuple

import torch
import torch.nn as nn

# Avoid transformers probing TensorFlow in this environment.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

from transformers import MarianMTModel, MarianTokenizer

from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"


class MarianStylePositionEmbed(nn.Module):
    """PositionEmbed compatible with Marian forward behavior."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, embed_scale: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("position_table", torch.zeros(1, max_seq_len, d_model))
        self.embed_scale = embed_scale

    def forward(self, X: torch.Tensor):
        pos = self.position_table[:, : X.shape[1]].to(X.device)
        return self.embedding(X) * self.embed_scale + pos


def build_models() -> Tuple[MarianTokenizer, MarianMTModel, Transformer]:
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    marian_model = MarianMTModel.from_pretrained(MODEL_NAME).to(device).eval()

    d_model = marian_model.config.d_model
    head_num = marian_model.config.encoder_attention_heads
    ffn_hidden_dim = marian_model.config.encoder_ffn_dim
    vocab_size = marian_model.model.shared.weight.shape[0]

    my_transformer = Transformer(
        max_seq_len=512, d_model=d_model, head_num=head_num, ffn_hidden_dim=ffn_hidden_dim
    )
    embed_scale = math.sqrt(d_model) if marian_model.config.scale_embedding else 1.0
    my_transformer.position_embed = MarianStylePositionEmbed(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=512,
        embed_scale=embed_scale,
    )
    my_transformer.linear = nn.Linear(d_model, vocab_size, bias=False)
    # Output raw logits; generation should be based on logits.
    my_transformer.softmax = nn.Identity()

    for layer in my_transformer.encoder_layers:
        layer.FFN.relu = nn.SiLU()
        layer.LayerNorm1 = nn.LayerNorm(d_model, eps=1e-5)
        layer.LayerNorm2 = nn.LayerNorm(d_model, eps=1e-5)
    for layer in my_transformer.decoder_layers:
        layer.FFN.relu = nn.SiLU()
        layer.LayerNorm1 = nn.LayerNorm(d_model, eps=1e-5)
        layer.LayerNorm2 = nn.LayerNorm(d_model, eps=1e-5)
        layer.LayerNorm3 = nn.LayerNorm(d_model, eps=1e-5)

    return tokenizer, marian_model, my_transformer.to(device).eval()


def load_weights_from_marian(marian_model: MarianMTModel, my_transformer: Transformer) -> None:
    state_dict = marian_model.state_dict()
    with torch.no_grad():
        shared_emb = state_dict["model.shared.weight"]
        my_transformer.position_embed.embedding.weight.copy_(shared_emb)
        my_transformer.linear.weight.copy_(shared_emb)
        my_transformer.position_embed.position_table.copy_(
            state_dict["model.encoder.embed_positions.weight"].unsqueeze(0)
        )
        my_transformer.final_logits_bias = state_dict["final_logits_bias"].to(device)

        for i in range(6):
            enc_layer = my_transformer.encoder_layers[i]
            prefix = f"model.encoder.layers.{i}"
            q_w = state_dict[f"{prefix}.self_attn.q_proj.weight"]
            k_w = state_dict[f"{prefix}.self_attn.k_proj.weight"]
            v_w = state_dict[f"{prefix}.self_attn.v_proj.weight"]
            q_b = state_dict[f"{prefix}.self_attn.q_proj.bias"]
            k_b = state_dict[f"{prefix}.self_attn.k_proj.bias"]
            v_b = state_dict[f"{prefix}.self_attn.v_proj.bias"]

            enc_layer.MTA.self_linear.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            enc_layer.MTA.self_linear.bias.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            enc_layer.MTA.out_proj.weight.copy_(state_dict[f"{prefix}.self_attn.out_proj.weight"])
            enc_layer.MTA.out_proj.bias.copy_(state_dict[f"{prefix}.self_attn.out_proj.bias"])
            enc_layer.LayerNorm1.weight.copy_(state_dict[f"{prefix}.self_attn_layer_norm.weight"])
            enc_layer.LayerNorm1.bias.copy_(state_dict[f"{prefix}.self_attn_layer_norm.bias"])
            enc_layer.FFN.linear1.weight.copy_(state_dict[f"{prefix}.fc1.weight"])
            enc_layer.FFN.linear1.bias.copy_(state_dict[f"{prefix}.fc1.bias"])
            enc_layer.FFN.linear2.weight.copy_(state_dict[f"{prefix}.fc2.weight"])
            enc_layer.FFN.linear2.bias.copy_(state_dict[f"{prefix}.fc2.bias"])
            enc_layer.LayerNorm2.weight.copy_(state_dict[f"{prefix}.final_layer_norm.weight"])
            enc_layer.LayerNorm2.bias.copy_(state_dict[f"{prefix}.final_layer_norm.bias"])

        for i in range(6):
            dec_layer = my_transformer.decoder_layers[i]
            prefix = f"model.decoder.layers.{i}"
            q_w = state_dict[f"{prefix}.self_attn.q_proj.weight"]
            k_w = state_dict[f"{prefix}.self_attn.k_proj.weight"]
            v_w = state_dict[f"{prefix}.self_attn.v_proj.weight"]
            q_b = state_dict[f"{prefix}.self_attn.q_proj.bias"]
            k_b = state_dict[f"{prefix}.self_attn.k_proj.bias"]
            v_b = state_dict[f"{prefix}.self_attn.v_proj.bias"]

            dec_layer.MTA1.self_linear.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            dec_layer.MTA1.self_linear.bias.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            dec_layer.MTA1.out_proj.weight.copy_(state_dict[f"{prefix}.self_attn.out_proj.weight"])
            dec_layer.MTA1.out_proj.bias.copy_(state_dict[f"{prefix}.self_attn.out_proj.bias"])
            dec_layer.LayerNorm1.weight.copy_(state_dict[f"{prefix}.self_attn_layer_norm.weight"])
            dec_layer.LayerNorm1.bias.copy_(state_dict[f"{prefix}.self_attn_layer_norm.bias"])

            q_w = state_dict[f"{prefix}.encoder_attn.q_proj.weight"]
            k_w = state_dict[f"{prefix}.encoder_attn.k_proj.weight"]
            v_w = state_dict[f"{prefix}.encoder_attn.v_proj.weight"]
            q_b = state_dict[f"{prefix}.encoder_attn.q_proj.bias"]
            k_b = state_dict[f"{prefix}.encoder_attn.k_proj.bias"]
            v_b = state_dict[f"{prefix}.encoder_attn.v_proj.bias"]

            dec_layer.MTA2.self_linear.weight.copy_(q_w)
            dec_layer.MTA2.self_linear.bias.copy_(q_b)
            dec_layer.MTA2.other_linear.weight.copy_(torch.cat([k_w, v_w], dim=0))
            dec_layer.MTA2.other_linear.bias.copy_(torch.cat([k_b, v_b], dim=0))
            dec_layer.MTA2.out_proj.weight.copy_(state_dict[f"{prefix}.encoder_attn.out_proj.weight"])
            dec_layer.MTA2.out_proj.bias.copy_(state_dict[f"{prefix}.encoder_attn.out_proj.bias"])
            dec_layer.LayerNorm2.weight.copy_(state_dict[f"{prefix}.encoder_attn_layer_norm.weight"])
            dec_layer.LayerNorm2.bias.copy_(state_dict[f"{prefix}.encoder_attn_layer_norm.bias"])
            dec_layer.FFN.linear1.weight.copy_(state_dict[f"{prefix}.fc1.weight"])
            dec_layer.FFN.linear1.bias.copy_(state_dict[f"{prefix}.fc1.bias"])
            dec_layer.FFN.linear2.weight.copy_(state_dict[f"{prefix}.fc2.weight"])
            dec_layer.FFN.linear2.bias.copy_(state_dict[f"{prefix}.fc2.bias"])
            dec_layer.LayerNorm3.weight.copy_(state_dict[f"{prefix}.final_layer_norm.weight"])
            dec_layer.LayerNorm3.bias.copy_(state_dict[f"{prefix}.final_layer_norm.bias"])


def translate(text: str, tokenizer: MarianTokenizer, my_transformer: Transformer, max_length: int = 50):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], device=device)

    with torch.no_grad():
        for _ in range(max_length):
            seq_len = decoder_input_ids.shape[1]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            outputs = my_transformer(input_ids, decoder_input_ids, causal_mask)
            next_token = (outputs[:, -1, :] + my_transformer.final_logits_bias).argmax(dim=-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_tokens = decoder_input_ids[0][1:].cpu().tolist()
    translated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return translated_text


if __name__ == "__main__":
    tokenizer, marian_model, my_transformer = build_models()
    load_weights_from_marian(marian_model, my_transformer)
    for text in [
        "This is a great movie! I really enjoyed watching it.",
        "The acting was terrible, and the plot made no sense.",
        "Deep learning has completely revolutionized natural language processing.",
    ]:
        pred = translate(text, tokenizer, my_transformer)
        print(f"EN: {text}")
        print(f"DE: {pred}\n")