"""Public model patch entrypoints for SAKV."""
from cake.monkeypatch import (
    replace_flashllama_attn_with_sakv_attn,
    replace_flashmistral_attn_with_sakv_attn,
    replace_flashqwen2_attn_with_sakv_attn,
)
