"""
cross-attention
一般出现在(Encoder-Decoder / Seq2Seq)

常见场景为
机器翻译 source=英文句子 target=中文句子
摘要 source=长文章 target=摘要
任务型生成 source=指令/表格/代码片段 target=输出
传统 seq2seq(T5、BART 这类)


❌ 不需要 cross-attention(Decoder-only / GPT 类)
如果模型的“条件”已经直接放进了同一条序列里
prompt + answer 拼一起，那就不需要单独的 encoder / cross-attn

GPT [prompt tokens][answer tokens] 全在一个序列里

这时模型只靠 masked self-attention 就能“既看 prompt 又看已生成历史”
一句话：
如果条件信息已经在 decoder 的 token 序列里了，就不需要 encoder / cross-attn

关键
实际上encoder提供的是“条件信息”
encoder 的 self-attention 不需要 causal mask 
因为 source 是给定的、完整可见的
读文章当然可以看全文!!
"""

"""
Q 来自 decoder 上一层
x = [B, T_dec, C]

K,V 来自 encoder
enc_out = [B, T_enc, C]

Scores = T_tgt * T_src (每个 target token 看所有 source token 的相关性)
"""