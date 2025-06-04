
# ml-gpt2-ohlcv

A GPT‑2‑style language‑model pipeline for **financial time‑series**.  
Instead of predicting the *next word*, the model predicts the **next OHLCV “token”** for an instrument such as AAPL on a 5‑minute timeframe.

*Trained on ~24 M context‑window samples, the model reached the results below after three epochs.*
*The PnL was calculated from the aapl ticker, do note that the model was trained on crypto, forex, and on some minerals on various time frames but has never has seen the stock market.*

| Epoch | Train Loss | Val Loss |
|------:|-----------:|---------:|
| 1 | 4.0811 | 3.9624 |
| 2 | 3.9322 | 3.9153 |
| **3** | **3.8836** | **3.9028** |

## 3ᴿᴰ‑epoch highlights (AAPL M5, 2020‑04‑09 → 2025‑05‑23)

| Metric | Value |
|--------|-------|
| **Total simulated PnL** | **+$811.40** on 1 000 USD equity |
| Direction accuracy (all trades) | 55.13 % |
| Next‑token top‑1 accuracy | 13.07 % |
| Direction accuracy @ 50 % confidence | 77.02 % (only 0.83 % of cases) |
| Token accuracy @ 50 % confidence | 51.26 % |

> **Note** These are *research* numbers. Real‑world trading incurs latency, slippage and exchange fees; live performance would be lower.

## What this repo does
1. **Normalise** raw OHLCV CSVs with a 100‑bar, per‑row stochastic scaler.<br>
   *`normalize_market_data.py`*
2. **Vector‑quantise** each bar to a single integer token using a VQ‑VAE encoder / code‑book (2 048 tokens).<br>
   *`tokenize_normalized_data.py`, `train_vqvae.py`*
3. **Train** a GPT‑2 Small decoder‑only transformer on sliding windows of 100 tokens.<br>
   *`train.py`*
4. **Evaluate** confidence, coverage, direction, PnL and full inference metrics.<br>
   *`eval_conf.py`, `eval_trade_sim.py`, `infer.py`*

## Folder layout (important files)

```
data/
  market_data/         raw CSV/TSV OHLCV files
  norm/                normalised CSVs (NORM_*.csv)
  tokens/              integer‑token CSVs (TOKN_*.csv)

models/
  ohlcv_vqvae_encoder.pth   ↳ 2048‑code VQ‑VAE encoder
  vocab.json                ↳ token → avg [O,H,L,C,V]

network/                   GPT‑2 Time‑Series implementation
utils/                     helpers (scheduler, seed, save_ckpt)

train.py                   fine‑tunes GPT‑2 on tokens
eval_trade_sim.py          trade simulation
eval_conf.py               confidence vs accuracy curves
infer.py                   full classification & trend metrics
```

## Quick‑start

```bash
git clone https://github.com/rsenatorov/ml-gpt2-ohlcv.git
cd ml-gpt2-ohlcv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Prepare data**

   ```bash
   python src/normalize_market_data.py
   python src/tokenize_normalized_data.py
   ```

2. **Train**

   ```bash
   python src/train.py
   ```

   Checkpoints are written to `checkpoints/checkpointN.pth`.

3. **Evaluate**

   ```bash
   python src/eval_trade_sim.py   # PnL curve
   python src/eval_conf.py        # survival curve
   python src/infer.py            # classification metrics
   ```

## Reproducing the 3ᴿᴰ‑epoch results

With the provided AAPL M5 dataset (`data/test/market_data/AAPLUSUSD_M5.csv`)
and the default config:

```bash
python src/train.py          # 3 epochs → checkpoint3.pth
python src/eval_trade_sim.py # uses checkpoint3 by default
```

The script will log a **total profit of ≈ $811** and a **55 % direction hit‑rate**
across ~473 k trades on the test split.

## Limitations & disclaimer

- The model **does not model order‑book liquidity, spreads or fees**.  
- Past performance on historical candles is **not** indicative of live profits.  
- Use at your own risk; the author is **not liable** for financial losses.

## License

Released under the [MIT License](LICENSE) © 2025 **Robert Senatorov**.  
You may use, modify and redistribute the code for any purpose, provided you
include the original copyright notice.

## Citation

If you build on this work, please cite it:

```
@misc{ml-gpt2-ohlcv,
  author = {Robert Senatorov},
  title  = {ml-gpt2-ohlcv: GPT‑2 language model for OHLCV token prediction},
  year   = {2025},
  url    = {https://github.com/rsenatorov/ml-gpt2-ohlcv.git}
}
```
