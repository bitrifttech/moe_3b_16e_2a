# 3B‑param Tutel MoE Chatbot

Train a ~3B‑parameter GPT‑2‑style Top‑2 MoE on a single RTX 4060 16 GB card.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python train.py --use_deepspeed
```
