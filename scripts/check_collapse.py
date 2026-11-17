"""Offline collapse check: feed TRAINING frames through the finetuned policy,
measure predicted action span per dim vs demo span. No GPU needed on free runner? needs cuda."""
import argparse, json
import numpy as np
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    ds = LeRobotDataset(repo_id="libero_so101_expert", root=args.root)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}})
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    preds = []
    for i in idxs:
        row = ds[int(i)]
        frame = {"observation.state": np.asarray(row["observation.state"]),
                 "observation.images.camera1": np.asarray(row["observation.images.camera1"]),
                 "observation.images.camera2": np.asarray(row["observation.images.camera2"]),
                 "observation.images.camera3": np.asarray(row["observation.images.camera3"])}
        frame = prepare_observation_for_inference(frame, device, task=str(row.get("task", "")))
        frame = pre(frame)
        with torch.inference_mode():
            a = policy.select_action(frame)
        a = post(a)
        if isinstance(a, dict): a = a["action"]
        if isinstance(a, torch.Tensor): a = a.squeeze().cpu().numpy()
        preds.append(np.asarray(a).reshape(-1)[:6])
    preds = np.stack(preds)
    acts = np.asarray(ds.hf_dataset["action"])[:len(ds)]
    print("demo span per dim:  ", np.round(acts.max(0) - acts.min(0), 1))
    print("pred span per dim:  ", np.round(preds.max(0) - preds.min(0), 1))
    print("pred grip p5/p95:   ", np.round(np.percentile(preds[:, 5], [5, 50, 95]), 1))
    print("demo grip p5/p95:   ", np.round(np.percentile(acts[:, 5], [5, 50, 95]), 1))
if __name__ == "__main__":
    main()
