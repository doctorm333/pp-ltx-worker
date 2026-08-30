import base64, tempfile, traceback, torch, runpod
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PIPE = None

def load():
    global PIPE
    if PIPE is None:
        vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        PIPE = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
        try:
            PIPE.vae.enable_tiling()
        except Exception:
            pass
        PIPE.enable_model_cpu_offload()
    return PIPE

DEFAULT_NEG = ("bright colors, overexposed, static, blurred details, subtitles, worst quality, "
               "low quality, jpeg artifacts, ugly, deformed, watermark, text, low detail, flickering")

def handler(event):
    try:
        inp = event.get("input", {}) or {}
        prompt = (inp.get("prompt") or "a cinematic scene").strip()
        neg = inp.get("negative_prompt") or DEFAULT_NEG
        w = int(inp.get("width", 1280)); h = int(inp.get("height", 704))
        nf = int(inp.get("num_frames", 121)); fps = int(inp.get("fps", 24))
        steps = int(inp.get("num_inference_steps", 50))
        guidance = float(inp.get("guidance_scale", 5.0))
        seed = inp.get("seed")
        pipe = load()
        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed not in (None, "") else None
        frames = pipe(prompt=prompt, negative_prompt=neg, width=w, height=h,
                      num_frames=nf, num_inference_steps=steps, guidance_scale=guidance,
                      generator=gen).frames[0]
        out = tempfile.mktemp(suffix=".mp4")
        export_to_video(frames, out, fps=fps)
        with open(out, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {"video": "data:video/mp4;base64," + b64, "resolution": "%dx%d" % (w, h), "fps": fps}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()[-3500:]}

runpod.serverless.start({"handler": handler})
