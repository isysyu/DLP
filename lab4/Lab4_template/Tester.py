import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder
from torchvision.utils import save_image
from torch import stack
import pandas as pd
import imageio
from math import log10
from Trainer import VAE_Model
import glob

TA_ = """
 ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██████╗  █████╗ ████████╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗    ██╗██╗██╗
██╔════╝██╔═══██╗████╗  ██║██╔════╝ ██╔══██╗██╔══██╗╚══██╔══╝██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝    ██║██║██║
██║     ██║   ██║██╔██╗ ██║██║  ███╗██████╔╝███████║   ██║   ██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗    ██║██║██║
██║     ██║   ██║██║╚██╗██║██║   ██║██╔══██╗██╔══██║   ██║   ██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║    ╚═╝╚═╝╚═╝
╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║    ██╗██╗██╗
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ╚═╝╚═╝╚═╝                                                                                                                          
"""

def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)


from torch.utils.data import DataLoader
from glob import glob
from torch.utils.data import Dataset as torchData
from torchvision.datasets.folder import default_loader as imgloader


class Dataset_Dance(torchData):
    def __init__(self, root, transform, mode='test', video_len=7, partial=1.0):
        super().__init__()
        self.img_folder = []
        self.label_folder = []

        test_img_path = os.path.join(root, 'test', 'test_img')
        test_label_path = os.path.join(root, 'test', 'test_label')

        print(f"Looking for data in: {test_img_path}")
        if not os.path.exists(test_img_path):
            raise FileNotFoundError(f"Directory not found: {test_img_path}")

        data_num = len([name for name in os.listdir(test_img_path) if os.path.isdir(os.path.join(test_img_path, name))])
        print(f"Number of data folders found: {data_num}")

        for i in range(data_num):
            img_files = sorted(glob(os.path.join(test_img_path, str(i), '*')), key=get_key)
            label_files = sorted(glob(os.path.join(test_label_path, str(i), '*')), key=get_key)
            print(f"Folder {i}: {len(img_files)} images, {len(label_files)} labels")
            if len(img_files) > 0 and len(label_files) > 0:
                self.img_folder.append(img_files)
                self.label_folder.append(label_files)

        self.transform = transform
        print(f"Total number of sequences: {len(self.img_folder)}")

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, index):
        frame_seq = self.img_folder[index]
        label_seq = self.label_folder[index]

        imgs = []
        labels = []
        imgs.append(self.transform(imgloader(frame_seq[0])))
        for idx in range(len(label_seq)):
            labels.append(self.transform(imgloader(label_seq[idx])))
        return stack(imgs), stack(labels)


class Test_model(VAE_Model):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)

        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

    def forward(self, img, label):
        pass

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        print(f"Number of batches in val_loader: {len(val_loader)}")
        pred_seq_list = []
        for idx, (img, label) in enumerate(tqdm(val_loader, ncols=80)):
            print(f"Processing batch {idx}")
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            try:
                pred_seq = self.val_one_step(img, label, idx)
                print(f"Shape of pred_seq: {pred_seq.shape}")
                pred_seq_list.append(pred_seq)
            except Exception as e:
                print(f"Error in val_one_step: {e}")
        print(f"Number of sequences in pred_seq_list: {len(pred_seq_list)}")

        if len(pred_seq_list) == 0:
            print("pred_seq_list is empty")
            return

        # submission.csv is the file you should submit to kaggle
        pred_to_int = (np.rint(torch.cat(pred_seq_list).numpy() * 255)).astype(int)
        df = pd.DataFrame(pred_to_int)
        df.insert(0, 'id', range(0, len(df)))
        df.to_csv(os.path.join(self.args.save_root, f'submission.csv'), header=True, index=False)

    def val_one_step(self, img, label, idx=0):
        img = img.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4)  # change tensor into (seq, B, C, H, W)
        print(f"Input img shape: {img.shape}")
        print(f"Input label shape: {label.shape}")
        assert label.shape[0] == 630, "Testing pose seqence should be 630"
        assert img.shape[0] == 1, "Testing video seqence should be 1"

        decoded_frame_list = [img[0].cpu()]
        label_list = []

        out = img[0]

        for i in range(1, 630):
            f = self.frame_transformation(decoded_frame_list[i - 1].to(self.args.device))
            l = self.label_transformation(label[i]).to(self.args.device)
            z, _, _ = self.Gaussian_Predictor(f, l)
            z = torch.randn_like(z)
            decoded = self.Decoder_Fusion(f, l, z)
            decoded = self.Generator(decoded)
            decoded_frame_list.append(decoded.cpu())
            label_list.append(l.cpu())

        # Please do not modify this part, it is used for visulization
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)

        print(f"Generated frame shape: {generated_frame.shape}")
        assert generated_frame.shape == (1, 630, 3, 32,64), f"The shape of output should be (1, 630, 3, 32, 64), but your output shape is {generated_frame.shape}"

        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'pred_seq{idx}.gif'))

        generated_frame = generated_frame.reshape(630, -1)

        print(f"Shape of generated_frame: {generated_frame.shape}")
        return generated_frame

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(img_name, format="GIF", append_images=new_list,
                         save_all=True, duration=20, loop=0)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        print(f"Validation data root: {self.args.DR}")
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, video_len=self.val_vi_len)
        print(f"Dataset size: {len(dataset)}")
        val_loader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=self.args.num_workers,
                                drop_last=True,
                                shuffle=False)
        print(f"Validation loader size: {len(val_loader)}")
        return val_loader

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True)


def main(args):
    print(f"Dataset path: {args.DR}")
    print(f"Checkpoint path: {args.ckpt_path}")

    if not os.path.exists(args.DR):
        raise FileNotFoundError(f"Dataset directory not found: {args.DR}")

        # Check if the checkpoint file exists
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")

    os.makedirs(args.save_root, exist_ok=True)
    model = Test_model(args).to(args.device)
    model.load_checkpoint()
    model.eval()

    #used for lab4 demo
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"number of params: {pytorch_total_params}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["mps", "cpu"], default="mps")
    parser.add_argument('--optim', type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--no_sanity', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--make_gif', action='store_true')
    parser.add_argument('--DR', type=str, required=True, help="Your Dataset Path")
    parser.add_argument('--save_root', type=str, required=True, help="The path to save your data")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=70, help="number of total epoch")
    parser.add_argument('--per_save', type=int, default=3, help="Save checkpoint every seted epoch")
    parser.add_argument('--partial', type=float, default=1.0, help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len', type=int, default=16, help="Training video length")
    parser.add_argument('--val_vi_len', type=int, default=630, help="valdation video length")
    parser.add_argument('--frame_H', type=int, default=32, help="Height input image to be resize")
    parser.add_argument('--frame_W', type=int, default=64, help="Width input image to be resize")

    # Module parameters setting
    parser.add_argument('--F_dim', type=int, default=128, help="Dimension of feature human frame")
    parser.add_argument('--L_dim', type=int, default=32, help="Dimension of feature label frame")
    parser.add_argument('--N_dim', type=int, default=12, help="Dimension of the Noise")
    parser.add_argument('--D_out_dim', type=int, default=192, help="Dimension of the output in Decoder_Fusion")

    # Teacher Forcing strategy
    parser.add_argument('--tfr', type=float, default=1.0, help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde', type=int, default=10, help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step', type=float, default=0.1, help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path', type=str, default=None, help="The path of your checkpoints")

    # Training Strategy
    parser.add_argument('--fast_train', action='store_true')
    parser.add_argument('--fast_partial', type=float, default=0.4,
                        help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch', type=int, default=5, help="Number of epoch to use fast train mode")

    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type', type=str, default='Cyclical', help="")
    parser.add_argument('--kl_anneal_cycle', type=int, default=10, help="")
    parser.add_argument('--kl_anneal_ratio', type=float, default=1, help="")

    args = parser.parse_args()

    main(args)