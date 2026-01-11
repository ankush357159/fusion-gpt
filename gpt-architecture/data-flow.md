## LLM Application Data Flow

This document outlines the end-to-end lifecycle of a single user prompt, from the initial keystroke to the final rendered response.

### 1. User Interaction

* **Input:** User types *"Explain quantum computing"* and submits.

### 2. Frontend Layer (Next.js)

* **Validation:** Locally validates input length and content type.
* **Rate Limiting:** Checks client-side cache to prevent rapid-fire submissions.
* **Optimistic UI:** Immediately adds the user's message to the UI to provide instant feedback.
* **Connection:** Opens a **Server-Sent Events (SSE)** connection to `/v1/chat/completions`.

### 3. Edge & Gateway Layer

* **Edge (Cloudflare/AWS):**
* SSL/TLS termination.
* WAF inspection and DDoS protection.
* Routing to the geographically nearest healthy server.


* **API Gateway (NGINX/Kong):**
* **Auth:** Verifies JWT token (decoding and expiration check).
* **Rate Limit:** Checks Redis-backed counter (e.g., 100 req/hour per user).
* **Observability:** Injects a Correlation ID and logs the incoming request.



### 4. Backend Processing (FastAPI)

* **Request Schema:** Pydantic validation of the JSON payload.
* **Security Pipeline:**
* Prompt injection detection.
* PII (Personally Identifiable Information) masking.
* Safety/Content filtering.


* **Semantic Cache:** Checks Redis for a SHA256 hash of the prompt + parameters.
* *If Hit:* Return cached response immediately.
* *If Miss:* Proceed to inference.



### 5. Prompt Engineering & Queue

* **Chat Templating:** Applies the **Llama 3.1** format:
```text
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.
<|start_header_id|>user<|end_header_id|>
Explain quantum computing
<|start_header_id|>assistant<|end_header_id|>

```


* **Context Management:** Injects relevant conversation history for multi-turn dialogue.
* **Inference Queue:**
* Adds request to the batch queue (max wait: 50ms).
* Collects up to 8 requests for optimal GPU throughput.



### 6. Model Inference (vLLM Engine)

* **KV Cache:** Checks for common prefix matches to skip redundant calculations.
* **Weight Loading:** Utilizes **4-bit GPTQ** quantized weights for memory efficiency.
* **Auto-regressive Generation:**
* Predicts tokens sequentially (~20-50ms per token).
* Tokens are generated and streamed immediately (e.g., *"Quantum"* → *" computing"* → *" uses"*).



### 7. Streaming Response & Frontend Update

* **SSE Events:** FastAPI yields data chunks in real-time:
`data: {"text": "Quantum", "done": false}`
* **UI Re-rendering:** React receives chunks, appends them to `message.content`, and triggers a re-render.
* **Visual Effect:** The user sees the text appearing word-by-word (ChatGPT-style).

### 8. Completion & Cleanup

* **Final Event:** `data: {"text": "", "done": true, "tokens": 247}`.
* **Async Persistence:**
* Saves the full conversation to **PostgreSQL**.
* Stores the final response in **Redis** (TTL: 1 hour).


* **Telemetry:** Updates **Prometheus** metrics:
* `request_count++`
* `tokens_generated += 247`
* `latency.observe(2.3s)`


* **Termination:** SSE connection is closed.

### **Final Result**

**User sees a complete, high-quality response displayed with a smooth, low-latency streaming effect.**