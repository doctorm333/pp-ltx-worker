import os, base64, json, subprocess, traceback, tempfile, glob, urllib.request, runpod
import yaml

VOL = "/runpod-volume"
SCHNELL_PATH = os.path.join(VOL, "models", "FLUX.1-schnell")
ADAPTER = "ostris/FLUX.1-schnell-training-adapter"
APP_BASE_URL = os.environ.get("APP_BASE_URL", "")
LORA_SECRET = os.environ.get("LORA_UPLOAD_SECRET", "")

def build_config(name, trigger, ds_dir, out_dir, steps, sample_prompts, quantize=True):
    return {
        "job": "extension",
        "config": {"name": name, "process": [{
            "type": "sd_trainer", "training_folder": out_dir, "device": "cuda:0", "trigger_word": trigger,
            "network": {"type": "lora", "linear": 16, "linear_alpha": 16},
            "save": {"dtype": "float16", "save_every": steps, "max_step_saves_to_keep": 1, "push_to_hub": False},
            "datasets": [{"folder_path": ds_dir, "caption_ext": "txt", "caption_dropout_rate": 0.05,
                          "shuffle_tokens": False, "cache_latents_to_disk": True, "resolution": [512, 768, 1024]}],
            "train": {"batch_size": 1, "steps": steps, "gradient_accumulation_steps": 1, "train_unet": True,
                      "train_text_encoder": False, "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                      "optimizer": "adamw8bit", "lr": 1e-4, "skip_first_sample": True,
                      "ema_config": {"use_ema": True, "ema_decay": 0.99}, "dtype": "bf16"},
            "model": {"name_or_path": SCHNELL_PATH, "assistant_lora_path": ADAPTER, "is_flux": True, "quantize": quantize},
            "sample": {"sampler": "flowmatch", "sample_every": steps, "sample_start_step": 0, "width": 1024, "height": 1024,
                       "prompts": sample_prompts, "neg": "", "seed": 42, "walk_seed": True, "guidance_scale": 1, "sample_steps": 4},
        }]},
        "meta": {"name": "[name]", "version": "1.0"},
    }

def upload_lora(user, name, path):
    if not (APP_BASE_URL and LORA_SECRET and path and os.path.exists(path)):
        return "skip"
    try:
        data = open(path, "rb").read()
        req = urllib.request.Request("%s/api/lora/upload?user=%s&name=%s" % (APP_BASE_URL, user, name),
                                     data=data, method="POST",
                                     headers={"x-lora-secret": LORA_SECRET, "Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=240) as r:
            return "ok" if r.status == 200 else "http%s" % r.status
    except Exception as e:
        return "err:" + str(e)[:120]

def handler(event):
    try:
        inp = event.get("input", {}) or {}
        if inp.get("debug"):
            return {"vol": os.path.isdir(VOL), "schnell": os.path.isdir(SCHNELL_PATH),
                    "ai_toolkit": os.path.isdir("/ai-toolkit"), "app": bool(APP_BASE_URL and LORA_SECRET)}
        images = inp.get("images") or []
        trigger = inp.get("trigger_word", "TOK")
        name = inp.get("name", "testlora")
        user = inp.get("user_id", "test")
        steps = int(inp.get("steps", 1000))
        if not images:
            return {"error": "no images provided"}
        ds_dir = tempfile.mkdtemp(prefix="ds_")
        for i, img in enumerate(images):
            b = img.split(",", 1)[1] if isinstance(img, str) and img.startswith("data:") else img
            open(os.path.join(ds_dir, "%03d.jpg" % i), "wb").write(base64.b64decode(b))
            open(os.path.join(ds_dir, "%03d.txt" % i), "w").write("photo of %s person" % trigger)
        out_dir = os.path.join(VOL, "loras", user)
        os.makedirs(out_dir, exist_ok=True)
        prompts = inp.get("sample_prompts") or ["photo of %s person, professional headshot, studio lighting" % trigger]
        cfg = build_config(name, trigger, ds_dir, out_dir, steps, prompts, bool(inp.get("quantize", True)))
        cfg_path = os.path.join(tempfile.mkdtemp(), "cfg.yaml")
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
        p = subprocess.run(["python", "/ai-toolkit/run.py", cfg_path], capture_output=True, text=True,
                           timeout=int(inp.get("timeout", 3400)), env=dict(os.environ))
        found = glob.glob(os.path.join(out_dir, name, "*.safetensors")) + \
                glob.glob(os.path.join(out_dir, "**", "*.safetensors"), recursive=True)
        lora = found[0] if found else None
        up = upload_lora(user, name, lora) if lora else "no-lora"
        return {"lora_path": lora, "returncode": p.returncode, "uploaded": up, "stdout_tail": p.stdout[-1500:],
                "stderr_tail": p.stderr[-1000:] if p.returncode else ""}
    except subprocess.TimeoutExpired as e:
        so = e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else str(e.stdout or "")
        return {"error": "training timeout", "stdout_tail": so[-1500:]}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()[-3500:]}

runpod.serverless.start({"handler": handler})
