import base64, tempfile, traceback, torch, runpod
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

PIPE = None
def load():
    global PIPE
    if PIPE is None:
        PIPE = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
        try:
            PIPE.vae.enable_tiling()
        except Exception:
            pass
        PIPE.enable_model_cpu_offload()
    return PIPE

def handler(event):
    try:
        inp = event.get("input", {}) or {}
        prompt = (inp.get("prompt") or "a cinematic scene").strip()
        neg = inp.get("negative_prompt") or "worst quality, blurry, distorted, watermark, text"
        w = int(inp.get("width", 704)); h = int(inp.get("height", 480))
        nf = int(inp.get("num_frames", 81)); fps = int(inp.get("fps", 24))
        steps = int(inp.get("num_inference_steps", 25))
        seed = inp.get("seed")
        pipe = load()
        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed not in (None, "") else None
        frames = pipe(prompt=prompt, negative_prompt=neg, width=w, height=h,
                      num_frames=nf, num_inference_steps=steps, generator=gen).frames[0]
        out = tempfile.mktemp(suffix=".mp4")
        export_to_video(frames, out, fps=fps)
        with open(out, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {"video": "data:video/mp4;base64," + b64, "resolution": "%dx%d" % (w, h), "fps": fps}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()[-3500:]}

runpod.serverless.start({"handler": handler})
