import base64, io, os, traceback, urllib.request, torch, runpod
from diffusers import FluxPipeline
from huggingface_hub import snapshot_download

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
HF_TOKEN = os.environ.get("HF_TOKEN")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "")
VOL = "/runpod-volume"
LOCAL_DIR = os.path.join(VOL, "models", "FLUX.1-schnell")
PATTERNS = ['transformer/*','text_encoder/*','text_encoder_2/*','vae/*','scheduler/*','tokenizer/*','tokenizer_2/*','*.json','*.txt']
PIPE = None

def ensure_on_volume():
    done = os.path.join(LOCAL_DIR, ".complete")
    if os.path.exists(done):
        return LOCAL_DIR
    os.makedirs(LOCAL_DIR, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=LOCAL_DIR, token=HF_TOKEN, allow_patterns=PATTERNS)
    open(os.path.join(LOCAL_DIR, ".complete"), "w").write("ok")
    return LOCAL_DIR

def load():
    global PIPE
    if PIPE is None:
        src = ensure_on_volume() if os.path.isdir(VOL) else MODEL_ID
        PIPE = FluxPipeline.from_pretrained(src, torch_dtype=torch.bfloat16, token=HF_TOKEN)
        PIPE.enable_model_cpu_offload()
        try:
            PIPE.vae.enable_tiling()
        except Exception:
            pass
    return PIPE

def lora_path(user, name):
    # 1) локально на томе (где тренировалось)
    for p in (os.path.join(VOL, "loras", user, name, name + ".safetensors"),
              os.path.join(VOL, "loras", user, name + ".safetensors")):
        if os.path.exists(p):
            return p
    # 2) кэш прошлой загрузки из хранилища
    cache_root = os.path.join(VOL, "loras", "_cache") if os.path.isdir(VOL) else os.path.join("/tmp", "lora_cache")
    cp = os.path.join(cache_root, user, name + ".safetensors")
    if os.path.exists(cp) and os.path.getsize(cp) > 1000:
        return cp
    # 3) скачать из центрального хранилища приложения
    if APP_BASE_URL:
        try:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            url = "%s/api/lora/file?user=%s&name=%s" % (APP_BASE_URL, user, name)
            urllib.request.urlretrieve(url, cp)
            if os.path.exists(cp) and os.path.getsize(cp) > 1000:
                return cp
        except Exception:
            pass
    return None

def handler(event):
    try:
        inp = event.get("input", {}) or {}
        if inp.get("debug"):
            loras = {}
            root = os.path.join(VOL, "loras")
            if os.path.isdir(root):
                for u in os.listdir(root):
                    try: loras[u] = sorted(os.listdir(os.path.join(root, u)))
                    except Exception: pass
            return {"vol": os.path.isdir(VOL), "model_complete": os.path.exists(os.path.join(LOCAL_DIR, ".complete")),
                    "app": bool(APP_BASE_URL), "loras": loras}
        prompt = (inp.get("prompt") or "a photo").strip()
        w = int(inp.get("width", 1024)); h = int(inp.get("height", 1024))
        steps = int(inp.get("num_inference_steps", 4))
        seed = inp.get("seed")
        lora_user = inp.get("lora_user"); lora_name = inp.get("lora_name")
        lora_scale = float(inp.get("lora_scale", 1.0))
        pipe = load()
        applied = False
        if lora_user and lora_name:
            lp = lora_path(lora_user, lora_name)
            if not lp:
                return {"error": "lora not found: %s/%s" % (lora_user, lora_name)}
            pipe.load_lora_weights(lp, adapter_name="user")
            try:
                pipe.set_adapters(["user"], adapter_weights=[lora_scale])
            except Exception:
                pass
            applied = True
        try:
            gen = torch.Generator("cpu").manual_seed(int(seed)) if seed not in (None, "") else None
            img = pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=steps,
                       max_sequence_length=256, width=w, height=h, generator=gen).images[0]
        finally:
            if applied:
                try: pipe.unload_lora_weights()
                except Exception: pass
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"image": "data:image/png;base64," + b64, "resolution": "%dx%d" % (w, h), "lora": bool(applied)}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()[-3500:]}

runpod.serverless.start({"handler": handler})
