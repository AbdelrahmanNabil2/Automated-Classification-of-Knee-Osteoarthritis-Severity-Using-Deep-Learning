import glob
import json
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as vutils
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "Original Dataset train"
OUTPUT_DIR = ROOT_DIR / "local_outputs"
MODELS_DIR = OUTPUT_DIR / "models"
GENERATED_SAMPLES = OUTPUT_DIR / "generated_samples"
GENERATED_IMAGES = OUTPUT_DIR / "generated_images"

GRADE_LABELS = {
    0: "Grade 0 (No OA)",
    1: "Grade 1 (Doubtful OA)",
    2: "Grade 2 (Mild OA)",
    3: "Grade 3 (Moderate OA)",
    4: "Grade 4 (Severe OA)",
}

LATENT_DIM = 100
IMAGE_SIZE = 64
NUM_CHANNELS = 1
GEN_FEATURES = 64
DISC_FEATURES = 64
LEARNING_RATE = 0.0002
LEARNING_RATE_D = LEARNING_RATE
BETAS = (0.5, 0.999)
BATCH_SIZE = 64
CHECKPOINT_INTERVAL = 25
MAX_EPOCHS = 650
TOP_K = 5


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
seed_everything(42)

_PIN_MEMORY = device.type == "cuda"
if torch.backends.mps.is_available():
    try:
        torch.mps.manual_seed(42)
    except AttributeError:
        pass


def ensure_dirs():
    for p in [MODELS_DIR, GENERATED_SAMPLES, GENERATED_IMAGES]:
        p.mkdir(parents=True, exist_ok=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, GEN_FEATURES * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(GEN_FEATURES * 8, GEN_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(GEN_FEATURES * 4, GEN_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(GEN_FEATURES * 2, GEN_FEATURES, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURES),
            nn.ReLU(True),
            nn.ConvTranspose2d(GEN_FEATURES, NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = DISC_FEATURES
        self.main = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_dataloader(grade):
    if grade not in GRADE_LABELS:
        raise ValueError("grade must be 0..4")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=transform)
    class_name = str(grade)
    if class_name not in dataset.class_to_idx:
        raise FileNotFoundError(f"Class folder not found: {DATASET_DIR / class_name}")

    class_idx = dataset.class_to_idx[class_name]
    grade_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == class_idx]
    subset = Subset(dataset, grade_indices)

    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=_PIN_MEMORY,
        drop_last=False,
    )


def _torch_load(path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(generator, discriminator, optim_g, optim_d, epoch, grade):
    gen_path = MODELS_DIR / f"generator_grade{grade}_epoch{epoch}.pth"
    disc_path = MODELS_DIR / f"discriminator_grade{grade}_epoch{epoch}.pth"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": generator.state_dict(),
            "optimizer_state_dict": optim_g.state_dict(),
        },
        gen_path,
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": discriminator.state_dict(),
            "optimizer_state_dict": optim_d.state_dict(),
        },
        disc_path,
    )


def load_latest_checkpoint(generator, discriminator, optim_g, optim_d, grade):
    pattern = str(MODELS_DIR / f"generator_grade{grade}_epoch*.pth")
    cps = glob.glob(pattern)
    if not cps:
        return 0

    def ep(path):
        return int(Path(path).stem.split("epoch")[-1])

    latest = max(cps, key=ep)
    epoch = ep(latest)
    gen_path = MODELS_DIR / f"generator_grade{grade}_epoch{epoch}.pth"
    disc_path = MODELS_DIR / f"discriminator_grade{grade}_epoch{epoch}.pth"

    gen_ckpt = _torch_load(gen_path)
    disc_ckpt = _torch_load(disc_path)
    generator.load_state_dict(gen_ckpt["model_state_dict"])
    discriminator.load_state_dict(disc_ckpt["model_state_dict"])
    optim_g.load_state_dict(gen_ckpt["optimizer_state_dict"])
    optim_d.load_state_dict(disc_ckpt["optimizer_state_dict"])
    return epoch


def save_preview_grid(generator, save_path, title=None):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(25, LATENT_DIM, 1, 1, device=device)
        fake = generator(noise).detach().cpu()
    generator.train()

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    if title:
        fig.suptitle(title, fontsize=12)
    for i, ax in enumerate(axes.flat):
        img = fake[i].squeeze().numpy() * 0.5 + 0.5
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_best_generator(generator, grade, epoch):
    p = MODELS_DIR / f"generator_grade{grade}_best_epoch{epoch}.pth"
    torch.save({"epoch": epoch, "model_state_dict": generator.state_dict()}, p)
    return str(p)


def update_top5_records(top5, candidate):
    top5.append(candidate)
    top5.sort(key=lambda x: x["total_loss"])
    while len(top5) > TOP_K:
        removed = top5.pop(-1)
        for key in ["preview_path", "best_ckpt_path"]:
            rp = removed.get(key)
            if rp and os.path.exists(rp):
                os.remove(rp)
    return top5


NAME_RE = re.compile(r"^(\d{7})([LR])(?:\.[a-zA-Z]+)?$")


def next_dataset_style_name_factory(grade, output_dir):
    numbers = []
    sides_seen = set()

    class_dir = DATASET_DIR / str(grade)
    if class_dir.exists():
        for p in class_dir.iterdir():
            if p.is_file():
                m = NAME_RE.match(p.name)
                if m:
                    numbers.append(int(m.group(1)))
                    sides_seen.add(m.group(2))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        m = NAME_RE.match(p.name)
        if m:
            numbers.append(int(m.group(1)))

    start = max(numbers) + 1 if numbers else 9000000
    side_order = ["L", "R"] if "L" in sides_seen or "R" in sides_seen else ["L", "R"]
    state = {"num": start, "idx": 0}

    def _next_name():
        side = side_order[state["idx"] % 2]
        name = f"{state['num']:07d}{side}.png"
        state["idx"] += 1
        if state["idx"] % 2 == 0:
            state["num"] += 1
        return name

    return _next_name


def _generate_from_top5_for_grade(grade, top5_records=None):
    top5_json = MODELS_DIR / f"top5_losses_grade{grade}.json"

    if top5_records is None:
        if not top5_json.exists():
            print(f"No top-5 file found: {top5_json}")
            print("Train this grade first.")
            return
        with open(top5_json, "r") as f:
            top5_records = json.load(f)

    if not top5_records:
        print("Top-5 records are empty.")
        return

    print("\nTop-5 best epochs/models:")
    for i, rec in enumerate(top5_records, start=1):
        print(
            f"[{i}] Epoch {rec['epoch']} | D: {rec['d_loss']:.4f} | "
            f"G: {rec['g_loss']:.4f} | Total: {rec['total_loss']:.4f}"
        )
        print(f"    Preview: {rec['preview_path']}")

    fig, axes = plt.subplots(1, len(top5_records), figsize=(4 * len(top5_records), 4))
    if len(top5_records) == 1:
        axes = [axes]
    for idx, (ax, rec) in enumerate(zip(axes, top5_records), start=1):
        img = plt.imread(rec["preview_path"])
        ax.imshow(img)
        ax.set_title(f"#{idx} E{rec['epoch']}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    pick = int(input("Choose model rank (1-5): ").strip())
    if pick < 1 or pick > len(top5_records):
        print("Invalid selection.")
        return

    num_images = int(input("How many images to generate? ").strip())
    if num_images < 1:
        print("Must be >= 1")
        return

    selected = top5_records[pick - 1]
    ckpt_path = selected["best_ckpt_path"]
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint missing: {ckpt_path}")
        return

    generator = Generator().to(device)
    ckpt = _torch_load(ckpt_path)
    generator.load_state_dict(ckpt["model_state_dict"])
    generator.eval()

    out_dir = GENERATED_IMAGES / f"grade{grade}" / f"rank{pick}_epoch{selected['epoch']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    next_name = next_dataset_style_name_factory(grade, out_dir)

    preview = []
    saved = 0
    with torch.no_grad():
        while saved < num_images:
            cur = min(BATCH_SIZE, num_images - saved)
            noise = torch.randn(cur, LATENT_DIM, 1, 1, device=device)
            fake = generator(noise).cpu()
            for i in range(cur):
                img = fake[i] * 0.5 + 0.5
                file_name = next_name()
                vutils.save_image(img, out_dir / file_name)
                if len(preview) < 25:
                    preview.append(img)
                saved += 1

    print(f"Saved {saved} images to: {out_dir}")

    if preview:
        grid = vutils.make_grid(torch.stack(preview), nrow=5, normalize=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
        ax.set_title(f"Generated Preview - Grade {grade} | Rank {pick}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


def train_one_grade(grade):
    dataloader = get_dataloader(grade)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    optim_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    optim_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=BETAS)

    start_epoch = 0
    if glob.glob(str(MODELS_DIR / f"generator_grade{grade}_epoch*.pth")):
        print(f"Grade {grade}: existing checkpoints found.")
        r = input("Resume latest? (y/n): ").strip().lower()
        if r == "y":
            start_epoch = load_latest_checkpoint(generator, discriminator, optim_g, optim_d, grade)
            print(f"Grade {grade}: resumed from epoch {start_epoch}")

    print(f"\nDevice: {device}")
    print(f"Training grade {grade} | DCGAN 64x64 | {MAX_EPOCHS} epochs")

    top5_records = []
    d_losses, g_losses = [], []

    epoch_bar = tqdm(
        range(start_epoch + 1, MAX_EPOCHS + 1),
        desc=f"Grade {grade} Epochs",
        unit="epoch",
        position=0,
        leave=True,
    )

    for epoch in epoch_bar:
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches = 0

        batch_bar = tqdm(
            dataloader,
            desc=f"Grade {grade} - Epoch {epoch}/{MAX_EPOCHS}",
            unit="batch",
            position=1,
            leave=False,
        )

        for real_images, _ in batch_bar:
            real_images = real_images.to(device, non_blocking=_PIN_MEMORY)
            bs = real_images.size(0)

            discriminator.zero_grad(set_to_none=True)
            real_targets = torch.ones(bs, device=device)
            fake_targets = torch.zeros(bs, device=device)

            noise = torch.randn(bs, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            out_real = discriminator(real_images).view(-1)
            loss_real = criterion(out_real, real_targets)
            out_fake = discriminator(fake_images.detach()).view(-1)
            loss_fake = criterion(out_fake, fake_targets)
            loss_d = 0.5 * (loss_real + loss_fake)
            loss_d.backward()
            optim_d.step()

            generator.zero_grad(set_to_none=True)
            out_gen = discriminator(fake_images).view(-1)
            loss_g = criterion(out_gen, real_targets)
            loss_g.backward()
            optim_g.step()

            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()
            n_batches += 1

        avg_d = epoch_d_loss / max(1, n_batches)
        avg_g = epoch_g_loss / max(1, n_batches)
        total = avg_d + avg_g
        d_losses.append(avg_d)
        g_losses.append(avg_g)

        epoch_bar.set_postfix({"D": f"{avg_d:.4f}", "G": f"{avg_g:.4f}", "Total": f"{total:.4f}"})
        print(
            f"Grade {grade} | Epoch [{epoch}/{MAX_EPOCHS}] | "
            f"D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | Total: {total:.4f}"
        )


        should_add = (len(top5_records) < TOP_K) or (total < top5_records[-1]["total_loss"])
        if should_add:
            preview_dir = GENERATED_SAMPLES / f"grade{grade}" / "best_losses"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"best_epoch_{epoch}.png"
            save_preview_grid(generator, preview_path, title=f"Grade {grade} | Epoch {epoch} | Loss {total:.4f}")
            best_ckpt_path = save_best_generator(generator, grade, epoch)
            top5_records = update_top5_records(
                top5_records,
                {
                    "epoch": epoch,
                    "d_loss": float(avg_d),
                    "g_loss": float(avg_g),
                    "total_loss": float(total),
                    "preview_path": str(preview_path),
                    "best_ckpt_path": str(best_ckpt_path),
                },
            )


        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(generator, discriminator, optim_g, optim_d, epoch, grade)
            sample_dir = GENERATED_SAMPLES / f"grade{grade}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / f"epoch_{epoch}.png"
            save_preview_grid(generator, sample_path, title=f"Grade {grade} | Epoch {epoch}")

            img = plt.imread(sample_path)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Generated Samples - Grade {grade} - Epoch {epoch}")
            ax.axis("off")
            plt.tight_layout()
            plt.show()

            print(f"Epoch {epoch} checkpoint completed - auto continuing training...")

    top5_records.sort(key=lambda x: x["total_loss"])
    top5_json = MODELS_DIR / f"top5_losses_grade{grade}.json"
    with open(top5_json, "w") as f:
        json.dump(top5_records, f, indent=2)

    print(f"\nTop 5 best epochs/models for grade {grade}:")
    for i, rec in enumerate(top5_records, start=1):
        print(
            f"{i}. Epoch {rec['epoch']} | D: {rec['d_loss']:.4f} | "
            f"G: {rec['g_loss']:.4f} | Total: {rec['total_loss']:.4f}"
        )
        print(f"   Preview: {rec['preview_path']}")

    fig, axes = plt.subplots(1, len(top5_records), figsize=(4 * len(top5_records), 4))
    if len(top5_records) == 1:
        axes = [axes]
    for ax, rec in zip(axes, top5_records):
        img = plt.imread(rec["preview_path"])
        ax.imshow(img)
        ax.set_title(f"E{rec['epoch']}\nL={rec['total_loss']:.3f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d_losses, label="D Loss")
    ax.plot(g_losses, label="G Loss")
    ax.set_title(f"Training Curves - Grade {grade}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Training complete for grade {grade}. Top-5 metadata: {top5_json}")

    generate_now = input("\nGenerate now from these top-5 models? (y/n): ").strip().lower()
    if generate_now == "y":
        _generate_from_top5_for_grade(grade, top5_records=top5_records)


def train_gan():
    print("\nSelect class/grade to train:")
    for k, v in GRADE_LABELS.items():
        print(f"  {k} - {v}")

    grade = int(input("Enter grade to train (0-4): ").strip())
    if grade not in GRADE_LABELS:
        print("Invalid grade. Aborting.")
        return

    print("\n" + "=" * 60)
    print(f"START TRAINING GRADE {grade} - {GRADE_LABELS[grade]}")
    print("=" * 60)
    train_one_grade(grade)


def generate_from_top5():
    print("\nSelect class/grade to generate from:")
    for k, v in GRADE_LABELS.items():
        print(f"  {k} - {v}")
    grade = int(input("Enter grade (0-4): ").strip())

    _generate_from_top5_for_grade(grade)


def main():
    ensure_dirs()
    print("==================================================")
    print(" DCGAN Local (Mac M2) - Top-5 losses")
    print(f" Device: {device}")
    print("==================================================")
    print("1 - Train GAN (choose class)")
    print("2 - Generate from top-5 best models")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        train_gan()
    elif choice == "2":
        generate_from_top5()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
