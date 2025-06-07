python \
  -m vllm.entrypoints.openai.api_server \
  --model ~/data/llm-models/Qwen3-14B-AWQ \
  --served-model-name qwen3-14b-diy \
  --gpu-memory-utilization 0.55 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --host 0.0.0.0 \
  --port 8000

