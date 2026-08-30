import base64, io, os, traceback, torch, runpod
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
HF_TOKEN = os.environ.get("HF_TOKEN")
PIPE = None

def load():
    global PIPE
    if PIPE is None:
        PIPE = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, token=HF_TOKEN)
        PIPE.enable_model_cpu_offload()
        try:
            PIPE.vae.enable_tiling()
        except Exception:
            pass
    return PIPE

def handler(event):
    try:
        inp = event.get("input", {}) or {}
        prompt = (inp.get("prompt") or "a photo").strip()
        w = int(inp.get("width", 1024)); h = int(inp.get("height", 1024))
        steps = int(inp.get("num_inference_steps", 4))
        seed = inp.get("seed")
        pipe = load()
        gen = torch.Generator("cpu").manual_seed(int(seed)) if seed not in (None, "") else None
        img = pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=steps,
                   max_sequence_length=256, width=w, height=h, generator=gen).images[0]
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"image": "data:image/png;base64," + b64, "resolution": "%dx%d" % (w, h)}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()[-3500:]}

runpod.serverless.start({"handler": handler})
