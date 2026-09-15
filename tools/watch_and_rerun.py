"""Watch for checkpoint update, then run eval and re-run inference+GradCAM for a target image.

Usage: run this with the project venv Python in background (async terminal).
"""
import time
import sys
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "resnet18.yaml"
IMG_REL = Path("tomato-leaf-isolated-on-white-260nw-1167806389.webp")

def load_cfg(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run(cmd):
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode

def main():
    cfg = load_cfg(CFG_PATH)
    exp = cfg.get("experiment", {}).get("name", "resnet18_baseline")
    ckpt = ROOT / "reports" / "checkpoints" / f"{exp}_best.pt"

    print(f"Watching checkpoint: {ckpt}")
    initial = ckpt.stat().st_mtime if ckpt.exists() else None

    # poll until mtime changes
    while True:
        if ckpt.exists():
            mtime = ckpt.stat().st_mtime
            if initial is None or mtime > initial:
                print("Checkpoint updated at", time.ctime(mtime))
                break
        else:
            print("Checkpoint not present yet; waiting...")
        time.sleep(10)

    # run evaluation
    print("Running evaluation...")
    run([sys.executable, "-u", "-m", "src.eval", "--config", str(CFG_PATH)])

    # run inference+gradcam script inline using python -c to avoid external dependencies
    print("Running inference + Grad-CAM on target image...")
    script = r"""
from pathlib import Path
from PIL import Image
import torch
import src.inference as inference
import src.explain.gradcam as gradcam
from src.utils import project_root

root = project_root()
img = root / 'tomato-leaf-isolated-on-white-260nw-1167806389.webp'
if not img.exists():
    print('image not found', img); raise SystemExit(2)

cfg_path = str(root / 'configs' / 'resnet18.yaml')
model, cfg, label_names, device, ckpt_path = inference.load_model_for_inference(cfg_path)
pil = Image.open(img).convert('RGB')
img_size = int(cfg.get('data', {}).get('img_size', 224))
res = inference.predict_image(model, pil, device, img_size=img_size, label_names=label_names, top_k=3)
print('PREDICTIONS:')
for p in res['predictions']:
    print(p)

geom = gradcam.make_eval_geometry(img_size=img_size)
to_tensor_norm = gradcam.make_tensor_normalize()
pil_aligned = geom(pil)
x = to_tensor_norm(pil_aligned).unsqueeze(0).to(device)
for p in model.parameters(): p.requires_grad_(True)
model_name = cfg['model']['name']
target_layer = gradcam.get_module_by_name(model, gradcam.default_target_layer_name(model_name))
cam = gradcam.GradCAM(model, target_layer)
with torch.enable_grad():
    result = cam(x, target_idx=None)
overlay = gradcam.overlay_heatmap_on_image(pil_aligned, result.heatmap, alpha=0.45)
out_dir = root / 'reports' / 'figures' / 'gradcam' / 'user_images'
out_dir.mkdir(parents=True, exist_ok=True)
safe_label = res['top_prediction']['label'].replace('/','_')
out_path = out_dir / f"{img.stem}_post_retrain_pred-{safe_label}_{res['top_prediction']['confidence']:.3f}.png"
overlay.save(out_path)
print('Saved overlay:', out_path)
"""

    run([sys.executable, "-u", "-c", script])
    print("Watcher finished.")

if __name__ == '__main__':
    main()
