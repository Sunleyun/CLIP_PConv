import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.shiq_lmdb_dataset import SHIQLmdbDataset
from models.pconv_unet import PConvUNet
from losses.losses import make_loss_for_stage
from utils.metrics import psnr, psnr_masked
from utils.vis import save_debug_grid

def build_loader(lmdb_dir, split, batch_size, num_workers, use_g=True):
    ds = SHIQLmdbDataset(
        lmdb_dir=lmdb_dir,
        split=split,
        out_size=256,
        use_g=use_g,
        g_params=(1, 3, 3.0),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=(split == "train"),
    )

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_all = []
    psnr_boundary = []

    for batch in loader:
        I_in = batch["I_in"].to(device, non_blocking=True)
        I_gt = batch["I_gt"].to(device, non_blocking=True)
        M = batch["M"].to(device, non_blocking=True)
        G = batch["G"].to(device, non_blocking=True)
        x = batch["x"].to(device, non_blocking=True)

        valid = 1.0 - M  # treat highlight as hole
        pred = model(x, valid).clamp(0, 1)

        psnr_all.append(psnr(pred, I_gt))

        # boundary region metric: use G > 0.5 as mask
        boundary = (G > 0.5).float()
        psnr_boundary.append(psnr_masked(pred, I_gt, boundary))

    return float(sum(psnr_all) / max(len(psnr_all), 1)), float(sum(psnr_boundary) / max(len(psnr_boundary), 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_lmdb", type=str, required=True)
    ap.add_argument("--val_lmdb", type=str, required=True)
    ap.add_argument("--stage", type=str, default="stage3", choices=["stage1","stage2","stage3"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--logdir", type=str, default="runs/shiq")
    ap.add_argument("--outdir", type=str, default="outputs/shiq")
    ap.add_argument("--vis_every", type=int, default=500)     # steps
    ap.add_argument("--val_every", type=int, default=1)       # epochs
    ap.add_argument("--save_every", type=int, default=1)      # epochs
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    writer = SummaryWriter(args.logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(args.train_lmdb, "train", args.batch_size, args.num_workers, use_g=True)
    val_loader = build_loader(args.val_lmdb, "val", args.batch_size, max(2, args.num_workers//2), use_g=True)

    model = PConvUNet(in_ch=5, base=64, use_gn=True).to(device)
    criterion = make_loss_for_stage(args.stage)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp))

    best_val = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)

        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            I_in = batch["I_in"].to(device, non_blocking=True)
            I_gt = batch["I_gt"].to(device, non_blocking=True)
            M = batch["M"].to(device, non_blocking=True)
            G = batch["G"].to(device, non_blocking=True)

            valid = 1.0 - M  # 1=valid region, 0=hole(highlight)
            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(not args.no_amp)):
                pred = model(x, valid)
                losses = criterion(pred, I_gt, I_in, M, G)
                loss = losses["total"]

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            # Logging
            writer.add_scalar("train/total", losses["total"].item(), global_step)
            writer.add_scalar("train/l1", losses["l1"].item(), global_step)
            writer.add_scalar("train/ssim", losses["ssim"].item(), global_step)
            writer.add_scalar("train/grad_g", losses["grad_g"].item(), global_step)
            writer.add_scalar("train/lap_g", losses["lap_g"].item(), global_step)
            writer.add_scalar("train/keep", losses["keep"].item(), global_step)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Periodic visualization
            if global_step % args.vis_every == 0:
                vis_path = os.path.join(args.outdir, "vis", f"ep{epoch:03d}_step{global_step:06d}.png")
                save_debug_grid(vis_path, I_in, I_gt, pred.clamp(0,1), M, G, n=4)
                writer.add_image("debug/grid", torch.from_numpy(
                    __import__("cv2").cvtColor(__import__("cv2").imread(vis_path), __import__("cv2").COLOR_BGR2RGB)
                ).permute(2,0,1), global_step)

            global_step += 1

        # Validation
        if epoch % args.val_every == 0:
            val_psnr, val_psnr_b = validate(model, val_loader, device)
            writer.add_scalar("val/psnr", val_psnr, epoch)
            writer.add_scalar("val/psnr_boundary", val_psnr_b, epoch)

            # Save best
            if val_psnr > best_val:
                best_val = val_psnr
                ckpt_path = os.path.join(args.outdir, "checkpoints", "best.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "stage": args.stage,
                    "best_val_psnr": best_val,
                }, ckpt_path)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.outdir, "checkpoints", f"epoch_{epoch:03d}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "stage": args.stage,
                "best_val_psnr": best_val,
            }, ckpt_path)

    writer.close()
    print(f"Done. Best val PSNR = {best_val:.4f}")

if __name__ == "__main__":
    main()
