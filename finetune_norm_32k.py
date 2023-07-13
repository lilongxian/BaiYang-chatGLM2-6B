"""
CreateTime: 2023-07-11
Author: li-long·BaiYang
Description: 构建弹性数据标准化旋转位置词嵌入编码器
参考：
 @misc{longchat2023,
    title = {How Long Can Open-Source LLMs Truly Promise on Context Length?},
    url = {https://lmsys.org/blog/2023-06-29-longchat},
    author = {Dacheng Li*, Rulin Shao*, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang},
    month = {June},
    year = {2023}
}
"""
# Need to call this before importing transformers.
from rotary_position_emb_patch import replace_glm2_Rotary_Emb
replace_glm2_Rotary_Emb()

from finetuning import main

if __name__ == "__main__":
    main()
