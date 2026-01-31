# Mistral-Core

An implementation of the Mistral-7B-Instruct-v0.2 architecture utilizing standard pretrained weights for high-performance inference. This model functions as a foundational zero-shot reasoning engine without any additional local training or fine-tuning.

## Directory Structure

```
.
├─ .env.example
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ src
	 └─ vera_gpt
			├─ __init__.py
			├─ cli.py
			├─ config.py
			├─ model.py
			└─ server.py
```

## Setup

1. Create a virtual environment.
2. Install dependencies from requirements.txt.
3. Copy .env.example to .env and set HF_TOKEN if needed.

## Run (API)

Start the API server:

```
uvicorn vera_gpt.server:app --host 0.0.0.0 --port 8000
```

Example request:

```
POST /chat
{
	"messages": [
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "Explain zero-shot reasoning."}
	],
	"max_new_tokens": 200,
	"temperature": 0.7,
	"top_p": 0.95
}
```

## Run (CLI)

```
python -m vera_gpt.cli "Explain zero-shot reasoning."
```
