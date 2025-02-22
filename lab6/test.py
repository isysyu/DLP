import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler

from utils import set_random_seed
from dataset.dataloader import IclevrDataset
from model.DDPM import DDPM_NoisePredictor
from evaluate import evaluate_DDPM, show_DDPM_denoising_process

def test_DDPM(args, ddpm, ddpm_scheduler):
    for mode in ["test", "new_test"]:
        test_dataset = IclevrDataset(
            mode=mode,
            json_root=args.data_path,
            image_root=os.path.join(args.data_path, "iclevr"),
            num_cpus=8,
        )

        accuracy, generated_images = evaluate_DDPM(
            args, ddpm, ddpm_scheduler, test_dataset
        )

        print("---------------------------------")
        print(f"Accuracy for {mode}: {round(accuracy * 100, 2)}%")

        image_visualizations = make_grid(generated_images, nrow=8)

        save_image(image_visualizations, f"{args.output_dir}/{mode}_ddpm_result.png")


    denoising_process_images = show_DDPM_denoising_process(args, ddpm, ddpm_scheduler)

    denoising_process_images_grid = make_grid(
        denoising_process_images, nrow=len(denoising_process_images)
    )

    save_image(
        denoising_process_images_grid, f"{args.output_dir}/ddpm_denoising_process.png"
    )

def main(args):
    try:
        ddpm = DDPM_NoisePredictor(args.n_classes).to(args.device)

        print(f"Loading checkpoint from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device)

        print("Checkpoint keys:", checkpoint.keys())

        if 'ddpm' in checkpoint:
            ddpm.load_state_dict(checkpoint['ddpm'])
            print("Loaded DDPM state from checkpoint.")
        else:
            ddpm.load_state_dict(checkpoint)
            print("Loaded DDPM state directly from checkpoint.")

        if 'ddpm_scheduler' in checkpoint:
            ddpm_scheduler = checkpoint['ddpm_scheduler']
            print("Loaded scheduler from checkpoint.")
        else:
            ddpm_scheduler = DDPMScheduler(
                num_train_timesteps=args.num_train_timesteps,
                beta_schedule="squaredcos_cap_v2",
            )
            print("Created new scheduler.")

        test_DDPM(args, ddpm, ddpm_scheduler)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(
        description="DDPM with iclevr Dataset"
    )

    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--data_path", type=str, default="data", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_classes", type=int, default=24, help="Number of object label classes")
    parser.add_argument("--num_train_timesteps", type=int, default=300, help="Number of training timesteps")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.seed is not None:
        set_random_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)


#用以下指令執行test.py
#python test.py --model_path "your path" --output_dir "test_results" --num_train_timesteps 300
