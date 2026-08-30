import base64, io, os, traceback, torch, runpod
from diffusers import ZImagePipeline
from huggingface_hub import snapshot_download

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
VOL = "/runpod-volume"
LOCAL_DIR = os.path.join(VOL, "models", "Z-Image-Turbo")
PATTERNS = ['transformer/*','text_encoder/*','vae/*','scheduler/*','tokenizer/*','*.json','*.txt']
PIPE = None

def ensure_on_volume():
    done = os.path.join(LOCAL_DIR, ".complete")
    if os.path.exists(done):
        return LOCAL_DIR
    os.makedirs(LOCAL_DIR, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=LOCAL_DIR, allow_patterns=PATTERNS)
    open(os.path.join(LOCAL_DIR, ".complete"), "w").write("ok")
    return LOCAL_DIR

def load():
    global PIPE
    if PIPE is None:
        src = ensure_on_volume() if os.path.isdir(VOL) else MODEL_ID
        PIPE = ZImagePipeline.from_pretrained(src, torch_dtype=torch.bfloat16)
        PIPE.enable_model_cpu_offload()
        try:
            PIPE.vae.enable_tiling()
        except Exception:
            pass
    return PIPE

def handler(event):
    try:
        inp = event.get("input", {}) or {}
        if inp.get("debug"):
            return {"vol": os.path.isdir(VOL), "model_complete": os.path.exists(os.path.join(LOCAL_DIR, ".complete"))}
        prompt = (inp.get("prompt") or "a photo").strip()
        w = int(inp.get("width", 1024)); h = int(inp.get("height", 1024))
        steps = int(inp.get("num_inference_steps", 9))
        guidance = float(inp.get("guidance_scale", 0.0))
        seed = inp.get("seed")
        pipe = load()
        gen = torch.Generator("cpu").manual_seed(int(seed)) if seed not in (None, "") else None
        kw = dict(prompt=prompt, width=w, height=h, num_inference_steps=steps, guidance_scale=guidance, generator=gen)
        neg = inp.get("negative_prompt")
        if neg:
            kw["negative_prompt"] = neg
        img = pipe(**kw).images[0]
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "resolution": "%dx%d" % (w, h)}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()[-3500:]}

runpod.serverless.start({"handler": handler})
