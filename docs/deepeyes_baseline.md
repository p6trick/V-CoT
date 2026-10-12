# DeepEyes-7B clean ScienceQA baseline

This runner keeps the DeepEyes visual-reasoning protocol: the model is served
with vLLM, while the client reproduces the official `eval/eval_vstar.py` loop.
Each `<tool_call>` for `image_zoom_in_tool` crops the original image and passes
the crop back to the next model turn. It is not one-shot Qwen2.5-VL inference.

## Compatibility decision

Use Python **3.10**. The upstream DeepEyes vLLM 0.8 Dockerfile uses Python
3.10 and pins the matching family `vllm==0.8.2`, `torch==2.6.0`,
`torchvision==0.21.0`, and `transformers==4.51.3`. The upstream full
`requirements_deepeyes.txt` additionally contains RL training components;
this baseline intentionally does not install VeRL, Ray, DeepSpeed, or W&B.

## Local development (no model or dataset download)

```bash
cd /hdd2/Six/V-CoT
python3.10 -m venv .venv/deepeyes
source .venv/deepeyes/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/dev.txt
PYTHONPATH=src pytest -q
```

The local environment is only the CPU-side evaluation client and tests. Do not
install `requirements/deepeyes-server.txt`, start vLLM, or call the CLI locally.

## GPU server setup

```bash
git pull
python3.10 -m venv .venv/deepeyes
source .venv/deepeyes/bin/activate
python -m pip install --upgrade pip
# Select the PyTorch CUDA wheel matching the server before the remaining pins.
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/deepeyes-server.txt
python -m pip install -r requirements/deepeyes-client.txt

vllm serve ChenShawn/DeepEyes-7B \
  --trust-remote-code --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85
```

In another terminal, after the vLLM health check succeeds:

```bash
source .venv/deepeyes/bin/activate
PYTHONPATH=src python -m vcot_deepeyes.run_scienceqa \
  --config configs/deepeyes/scienceqa_clean.yaml
```

`streaming: true` reads the ScienceQA `test` split incrementally and the
loader emits only records whose `image` field is present. `max_samples: 20`
can be changed to 10 for the first smoke experiment. Results are JSONL under
`outputs/` (git-ignored); every line stores question, gold answer, final and
raw model response, accumulated generated-token count when vLLM reports it,
reasoning turns, successful zoom calls, latency, and the complete text/tool
trace. Crop artifacts are stored beside that sample and referenced from trace.

## Required server information before first run

Confirm GPU model(s) and VRAM, NVIDIA driver/CUDA version (`nvidia-smi`), OS,
available disk path for Hugging Face cache, outbound Hugging Face access or a
pre-staged checkpoint, and whether the vLLM server must bind beyond localhost.
For a 7B BF16 model plus a 32k context window, use a GPU with enough headroom
for weights and KV cache; adjust `max-model-len` and memory utilization only
after recording the hardware details.
