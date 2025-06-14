# 3B‑param Tutel MoE Chatbot

Train a ~3B‑parameter GPT‑2‑style Top‑2 MoE on a single RTX 4060 16 GB card.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python train.py --use_deepspeed
```

---

## Training: Start, Stop, and Resume

### Start Training

To start training from scratch (or resume from the latest checkpoint automatically):

```bash
python train.py --use_deepspeed
```
- Training will periodically save checkpoints in the `outputs/` directory (every 1000 steps by default).
- Logs and progress will be shown in the terminal and saved to `training.log`.

### Stop Training

You can safely stop training at any time (e.g., with `Ctrl+C` or by killing the process).
- All progress up to the last checkpoint will be saved.

### Resume Training

To resume training, simply run the same command again:

```bash
python train.py --use_deepspeed
```
- The script will automatically detect the latest checkpoint in `outputs/` and resume from there.
- If no checkpoint is found, training will start from scratch.

---

## Chat with Your Trained Model

After training, you can chat with your model using the provided `chat.py` script. This script will automatically load the latest checkpoint from the `outputs/` directory and start an interactive chat session.

### Usage

```bash
python chat.py
```

- The script will look for the latest checkpoint in `outputs/`.
- Type your message and press Enter to chat with the model.
- Type `exit` or `quit` to end the chat session.

---

## Using the Model for Chatting (Programmatically)

If you want to use the model in your own scripts, here is a simple example using the HuggingFace Transformers pipeline:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the latest checkpoint from outputs/
checkpoint_dir = 'outputs/checkpoint-<N>'  # Replace <N> with the latest checkpoint number

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

while True:
    prompt = input('You: ')
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('Bot:', response[len(prompt):].strip())
```

- Make sure to replace `checkpoint-<N>` with the actual latest checkpoint directory (e.g., `outputs/checkpoint-2000`).
- You can adapt this script for more advanced chat interfaces or batch inference.

---

## Notes
- On macOS ARM, only CPU training/inference is supported (much slower than GPU).
- Checkpoints and logs are saved in the `outputs/` directory by default.
- For custom training/evaluation settings, see `train.py` and adjust arguments as needed.
