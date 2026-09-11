# Key-driven Spectral State-matrix Codec（8 态）

**中文** | [English](README_EN.md)

把文字编成 8 态光谱矩阵；接收端识别每格状态后，用同一密钥还原。  
当前只公开 8 态代码。实测光谱、训练权重和论文数字不在仓库中（稿件准备中）。

自定义编解码，不能等同于 AES。`JLU_2026_Secret` 只是演示密钥。

## 运行

无需光谱：

```bash
pip install -r requirements.txt
python spectrum_codec_8_Secure.py
python step3_visualize_encoding_8.py
```

自备 8 类光谱时，格式见 [`dataset/README.md`](dataset/README.md)：

```bash
python step1_build_multimodal_dataset.py
python step2_train_multimodal_cnn_8class.py
python step4_end_to_end.py
```

密钥改写 3 bit→状态表，并打乱格子位置。识别状态与还原明文是两步。

## 许可

见 [LICENSE](LICENSE)。发表后补正式引用。
