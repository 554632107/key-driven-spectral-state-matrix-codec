# Key-driven Spectral State-matrix Codec (8-state)

[中文](README.md) | **English**

A message is written as an 8-state spectral matrix. After each cell is recognized, the same key restores the text.  
This release is the 8-state code only. Measured spectra, trained weights, and paper numbers are withheld (manuscript in preparation).

Custom codec, not equivalent to AES. `JLU_2026_Secret` is a demo key.

## Run

No spectra required:

```bash
pip install -r requirements.txt
python spectrum_codec_8_Secure.py
python step3_visualize_encoding_8.py
```

With your own 8-class spectra ([`dataset/README_EN.md`](dataset/README_EN.md)):

```bash
python step1_build_multimodal_dataset.py
python step2_train_multimodal_cnn_8class.py
python step4_end_to_end.py
```

The key rewrites the 3-bit-to-state table and scrambles cell positions. Recognition and plaintext recovery are separate steps.

## License

See [LICENSE](LICENSE). A formal citation will be added after publication.
