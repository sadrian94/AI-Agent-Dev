"""
test_local_llm.py
-----------------
Tests the local Ollama installation with 'gemma4:26b'.
Verifies GPU inference by inspecting the /api/ps endpoint to confirm
the model is loaded on VRAM rather than CPU RAM.
"""

import sys
import io
import time
import requests

# Force UTF-8 output on Windows so special characters don't crash the console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

OLLAMA_BASE   = "http://localhost:11434"
TARGET_MODEL  = "gemma4:26b"
PROMPT        = (
    "You are a helpful AI assistant. "
    "In 2–3 sentences, explain what makes a transformer architecture powerful for NLP tasks."
)

# ─────────────────────────────────────────────
# Helper: check GPU usage via Ollama REST API
# ─────────────────────────────────────────────
def check_gpu_usage(model_name: str) -> dict | None:
    """Query /api/ps and return the process entry for the running model."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
        r.raise_for_status()
        data = r.json()
        for proc in data.get("models", []):
            if proc.get("name", "").startswith(model_name.split(":")[0]):
                return proc
    except requests.RequestException as e:
        print(f"  [WARN] Could not reach Ollama REST API: {e}")
    return None


def print_gpu_info(proc: dict):
    """Pretty-print GPU/VRAM details from an /api/ps model entry."""
    size_vram = proc.get("size_vram", 0)
    size_total = proc.get("size", 0)
    details    = proc.get("details", {})

    print(f"\n  {'─'*42}")
    print(f"  Model loaded : {proc.get('name', 'unknown')}")
    print(f"  VRAM used    : {size_vram / 1e9:.2f} GB")
    print(f"  Total RAM    : {size_total / 1e9:.2f} GB")

    if size_vram > 0:
        pct = (size_vram / size_total * 100) if size_total else 0
        print(f"  GPU offload  : {pct:.1f}%  [GPU ACTIVE]")
    else:
        print("  GPU offload  :  0%  [WARN: CPU only]")

    if details:
        print(f"  Param size   : {details.get('parameter_size', 'N/A')}")
        print(f"  Quant level  : {details.get('quantization_level', 'N/A')}")
    print(f"  {'─'*42}\n")


# ─────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────
def test_ollama():
    print("=" * 50)
    print("  Ollama Local AI — GPU Inference Test")
    print(f"  Model  : {TARGET_MODEL}")
    print(f"  Server : {OLLAMA_BASE}")
    print("=" * 50)

    # 1. Ping server
    print("\n[1/4] Pinging Ollama server …")
    try:
        r = requests.get(f"{OLLAMA_BASE}", timeout=5)
        print(f"  [OK]  Server is up  (HTTP {r.status_code})")
    except requests.RequestException as e:
        print(f"  [ERR] Cannot reach Ollama: {e}")
        print("      -> Make sure Ollama Desktop / service is running.")
        return

    # 2. Build LLM client
    print("\n[2/4] Building ChatOllama client …")
    llm = ChatOllama(
        model=TARGET_MODEL,
        base_url=OLLAMA_BASE,
        temperature=0.3,
        num_gpu=99,          # ask Ollama to offload as many layers as possible to GPU
        num_ctx=4096,
    )
    print("  [OK]  Client ready")

    # 3. Run inference
    print(f"\n[3/4] Sending prompt to {TARGET_MODEL} …")
    messages = [
        SystemMessage(content="You are a concise technical assistant."),
        HumanMessage(content=PROMPT),
    ]

    t0 = time.perf_counter()
    try:
        response = llm.invoke(messages)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"  [ERR] Inference failed: {e}")
        return

    print(f"  [OK]  Response received in {elapsed:.2f}s")
    print("\n  ── Model Response ──────────────────────────")
    print(f"  {response.content.strip()}")
    print("  ────────────────────────────────────────────")

    # 4. GPU verification
    print("\n[4/4] Verifying GPU usage …")
    proc = check_gpu_usage(TARGET_MODEL)
    if proc:
        print_gpu_info(proc)
    else:
        print("  [WARN] No process info returned (model may have already unloaded).")
        print("         Re-run immediately after inference to catch the loaded state,")
        print("         or check 'ollama ps' in a terminal.")

    print("=" * 50)
    print("  Test complete.")
    print("=" * 50)


if __name__ == "__main__":
    test_ollama()
