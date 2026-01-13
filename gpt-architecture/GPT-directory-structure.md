## Directory structure


```
gpt-project/
│
├── configs/
│   ├── model.yaml
│   ├── training.yaml
│   ├── tokenizer.yaml
│   └── data.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── tokenized/
│   └── samples/
│
├── tokenizer/
│   ├── bpe.py
│   ├── vocab.json
│   └── merges.txt
│
├── model/
│   ├── gpt.py
│   ├── attention.py
│   ├── transformer.py
│   └── __init__.py
│
├── training/
│   ├── trainer.py
│   ├── losses.py
│   ├── optim.py
│   └── scheduler.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── benchmarks.py
│   └── generation_tests.py
│
├── inference/
│   ├── generate.py
│   ├── api.py
│   └── server.py
│
├── experiments/
│   ├── exp_001/
│   ├── exp_002/
│   └── logs/
│
├── checkpoints/
│   ├── best/
│   └── latest/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── tokenize.py
│   └── preprocess.py
│
├── monitoring/
│   ├── metrics.py
│   └── tensorboard/
│
├── tests/
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_training.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── README.md
└── .env
```

### Purpose of Each Layer

#### `configs/`

Centralized configuration:

* Model size
* Layers, heads
* Learning rate
* Batch size
* Tokenizer type

Enables **reproducible training**.

#### `data/`

Handles the full data pipeline:

* `raw/` → original C4, text files
* `processed/` → cleaned text
* `tokenized/` → BPE token IDs
* `samples/` → small debug sets

#### `tokenizer/`

Tokenizer logic:

* GPT-2 BPE
* tiktoken compatibility
* Vocabulary files

Keeps tokenization **decoupled** from model.

#### `model/`

Core GPT architecture:

* Multi-head attention
* Transformer blocks
* GPT forward pass

Reusable for training + inference.

#### `training/`

Training system:

* Optimizer
* LR scheduler
* Loss functions
* Gradient logic

Keeps training clean and testable.

#### `evaluation/`

Model quality:

* Perplexity
* Text generation tests
* Benchmarks

Ensures model improvements are measurable.

#### `inference/`

Production usage:

* Text generation
* REST API
* Serving logic

Supports real-world deployment.

#### `experiments/`

Tracks:

* Hyperparameter runs
* Logs
* Comparisons

Crucial for ML experimentation.

#### `checkpoints/`

Model versions:

* Best model
* Latest model
* Recovery points

Enables fault-tolerant training.

#### `scripts/`

Entry points:

* `train.py`
* `evaluate.py`
* `tokenize.py`

CLI-style execution.

#### `monitoring/`

Training observability:

* Loss curves
* GPU usage
* TensorBoard logs

Required for production readiness.

#### `tests/`

Unit tests:

* Model correctness
* Tokenizer consistency
* Training stability

Prevents silent failures.

#### `docker/`

Deployment environment:

* Reproducible builds
* Cloud compatibility