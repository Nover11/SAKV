import transformers
from sakv.model.modify_llama import llama_model_forward_sakv, llama_attn_forward_sakv
from sakv.model.modify_mistral import mistral_model_forward_sakv, mistral_attn_forward_sakv
from sakv.model.modify_qwen2 import qwen2_model_forward_sakv, qwen2_attn_forward_sakv
    

def replace_flashllama_attn_with_sakvattn():
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_sakv
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward_sakv

def replace_flashmistral_attn_with_sakeattn():
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_sakv
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_attn_forward_sakv

def replace_flashqwen2_attn_with_sakeattn():
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward_sakv
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_attn_forward_sakv

