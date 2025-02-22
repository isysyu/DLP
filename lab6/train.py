import os
import pytz
from tqdm import tqdm
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils import set_random_seed
from dataset.dataloader import IclevrDataset
from model.DDPM import DDPM_NoisePredictor
from evaluate import evaluate_DDPM

def train_DDPM(args):
    writer = SummaryWriter(log_dir=f"{args.output_dir}/logs")

    train_dataset = IclevrDataset(
        mode="train",
        json_root="data",
        image_root="data/iclevr",
        num_cpus=8,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    ddpm = DDPM_NoisePredictor(n_classes=args.n_classes).to(args.device)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_loader) * args.epochs),
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps, beta_schedule="squaredcos_cap_v2"
    )

    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        ddpm.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            batch_size = images.size(0)
            images, labels = images.to(args.device), labels.to(args.device)

            noises = torch.randn_like(images).to(args.device)

            timesteps = torch.randint(
                0,
                ddpm_scheduler.config.num_train_timesteps,
                (batch_size,),
                dtype=torch.int64,
            ).to(args.device)

            noisy_images = ddpm_scheduler.add_noise(images, noises, timesteps)

            pred_noises = ddpm(noisy_images, timesteps, labels)

            loss = criterion(pred_noises, noises)
            loss.backward()

            epoch_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_description(
                f"Epoch {epoch}/{args.epochs} [Batch {i}/{len(train_loader)}] [Loss: {loss.item():.4f}]"
            )

        writer.add_scalar("Loss/DDPM", epoch_loss / len(train_loader), epoch)

        if epoch % 2 == 0:
            with torch.no_grad():
                test_dataset = IclevrDataset(
                    mode="test",
                    json_root="data",
                    image_root="data/iclevr",
                    num_cpus=8,
                )
                accuracy, generated_images = evaluate_DDPM(
                    args, ddpm, ddpm_scheduler, test_dataset
                )

                image_visualizations = make_grid(generated_images, nrow=8)

                writer.add_scalar("Evaluation Accuracy", accuracy, epoch)
                writer.add_image("Generated Images", image_visualizations, epoch)

        if epoch % 10 == 0:
            torch.save(
                {
                    "ddpm": ddpm.state_dict(),
                    "ddpm_scheduler": ddpm_scheduler,
                },
                f"{args.output_dir}/ddpm_epoch{epoch}.pth",
            )
        elif epoch == args.epochs - 1:
            torch.save(
                {
                    "ddpm": ddpm.state_dict(),
                    "ddpm_scheduler": ddpm_scheduler,
                },
                f"{args.output_dir}/ddpm.pth",
            )

    print("Training completed!")

    for mode in ["test", "new_test"]:
        test_dataset = IclevrDataset(
            mode=mode,
            json_root="data",
            image_root="data/iclevr",
            num_cpus=8,
        )

        accuracy, generated_images = evaluate_DDPM(
            args, ddpm, ddpm_scheduler, test_dataset
        )

        print(f"Accuracy for {mode}: {accuracy}")

        image_visualizations = make_grid(generated_images, nrow=8)

        save_image(image_visualizations, f"{args.output_dir}/{mode}_ddpm_result.png")

def parse_args():
    parser = argparse.ArgumentParser(
        description="DDPM with iclevr Dataset"
    )

    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--data_path", type=str, default="data", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    #with num_workers = 8, it crash at 13
    #epoch = 30 with num_workers = 4 ->done, but accuracy isnt'well -> new_test =53
    parser.add_argument("--epochs", type=int, default=100)
    #batch_size = 128 is too large, 32->12min, 16->14.5
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--n_classes", type=int, default=24, help="Number of object label classes")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of training timesteps")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="Number of learning rate warmup steps")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.output_dir is not None:
        if args.debug:
            args.output_dir = f"{args.output_dir}/DDPM-debug"
        else:
            tz = pytz.timezone("Asia/Taipei")
            now = datetime.now(tz).strftime("%Y%m%d-%H%M")
            args.output_dir = f"{args.output_dir}/DDPM-{now}"

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/logs", exist_ok=True)

    print(f"Start training DDPM:")
    print(f"Output directory: {args.output_dir}\n")

    train_DDPM(args)


#直接執行便可執行train.py