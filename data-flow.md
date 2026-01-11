[Data Flow](data-flow.mmd)
User types "Explain quantum computing"
    ↓
[FRONTEND - Next.js]
├── Validate input locally (length, content)
├── Check client-side rate limit cache
├── Add message to UI optimistically (instant feedback)
└── Open SSE connection to /v1/chat/completions
    ↓
[EDGE LAYER - CDN/WAF]
├── SSL termination
├── DDoS protection
└── Route to nearest server
    ↓
[API GATEWAY - NGINX/Kong]
├── Verify JWT token (decode + check expiration)
├── Check Redis rate limit (100 req/hour per user)
├── Log request with correlation ID
└── Forward to backend
    ↓
[BACKEND - FastAPI]
├── Validate request schema (Pydantic)
├── Security checks:
│   ├── Prompt injection detection
│   ├── PII detection/masking
│   └── Content filtering
├── Check Redis cache (SHA256 of prompt + params)
│   └── If cache hit → return cached response
└── If cache miss → continue
    ↓
[PROMPT ENGINEERING]
├── Apply chat template (Llama 3.1 format):
│   <|start_header_id|>system<|end_header_id|>
│   You are a helpful assistant.
│   <|start_header_id|>user<|end_header_id|>
│   Explain quantum computing
│   <|start_header_id|>assistant<|end_header_id|>
└── Add conversation history if multi-turn
    ↓
[INFERENCE QUEUE]
├── Add to batch queue (wait max 50ms)
├── Collect up to 8 requests
└── Submit batch to vLLM
    ↓
[MODEL - vLLM Engine]
├── Check KV cache for common prefix
├── Load quantized weights (4-bit GPTQ)
├── Generate tokens auto-regressively:
│   Token 1: "Quantum" → stream
│   Token 2: "computing" → stream
│   Token 3: "uses" → stream
│   ... (continues for ~200-500 tokens)
└── Each token takes ~20-50ms
    ↓
[STREAMING BACK]
├── FastAPI sends SSE event: data: {"text": "Quantum", "done": false}
├── FastAPI sends SSE event: data: {"text": " computing", "done": false}
├── FastAPI sends SSE event: data: {"text": " uses", "done": false}
└── ... continues until completion
    ↓
[FRONTEND UPDATES]
├── Receive each SSE chunk
├── Append to message.content
├── Re-render component (React shows growing text)
└── User sees text appearing word-by-word (ChatGPT-style)
    ↓
[COMPLETION]
├── Send final SSE: data: {"text": "", "done": true, "tokens": 247}
├── Save conversation to PostgreSQL (async)
├── Cache response in Redis (TTL: 1 hour)
├── Update Prometheus metrics:
│   ├── request_count++
│   ├── tokens_generated += 247
│   └── request_latency.observe(2.3s)
└── Close SSE connection
    ↓
[USER SEES]
Complete response displayed with smooth streaming effect