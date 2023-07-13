"""
CreateTime: 2023-07-11
Author: li-long·BaiYang
Description: 构建数据标准化弹性旋转位置词嵌入编码器
参考：
 @misc{longchat2023,
    title = {How Long Can Open-Source LLMs Truly Promise on Context Length?},
    url = {https://lmsys.org/blog/2023-06-29-longchat},
    author = {Dacheng Li*, Rulin Shao*, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang},
    month = {June},
    year = {2023}
}
"""

import torch
from glm2_core import modeling_chatglm
from functools import partial


# NEW: 2023-07-11: 弹性数据标准化旋转位置词嵌入编码器
class NormRotaryEmbedding(torch.nn.Module):
    # P(p,2i) = sin(p/(10000^(2i/dim)))
    # P(p,2i+1) = cos(p/(10000^(2i/dim)))
    def __init__(self, dim, ratio=16, original_impl=False, device=None, dtype=None):
        super().__init__()
        """
        :param dim: chatGLM1和chatGLM2都使用了64 
        """
        # vector(32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.ratio = ratio
        self.original_impl = original_impl

    def forward_impl(self,
                     seq_len: int,  # config.seq_length: 32768
                     n_elem: int,  # 64
                     dtype: torch.dtype, device: torch.device,
                     base: int = 10000
                     ):
        """Enhanced Transformer with Rotary Position Embedding."""
        # 32个长度的向量for tensor. $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., 32767]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)/self.ratio

        # Calculate the product of position index and $\theta_i$
        # 外积：outer
        idx_theta = torch.einsum("i,j->ij", seq_idx, theta).float()  # [32768, 32]

        # R_POS[32768, 32, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        """
        :param max_seq_len: 32768 for GLM2 and 2048 for GLM1
        :param offset:
        :return: R_POS[32768, 32, 2]
        """
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


def replace_glm2_Rotary_Emb(ratio=16):
    """ 替换RotaryEmbedding类 """
    modeling_chatglm.RotaryEmbedding = partial(NormRotaryEmbedding, ratio=ratio)

