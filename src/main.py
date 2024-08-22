# Note: numpy 1.21 and scipy 1.3.3 were used. 

import GAN
from trainer import Trainer
from dataset import Dataset
from tensorboardX import SummaryWriter

from pytorch_fid import fid_score

import torch
import torch.optim as optim
import os
import argparse
from torchvision.utils import make_grid, save_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--generator_hidden_dim', default=16, type=int)
    parser.add_argument('--discriminator_hidden_dim', default=16, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_training_steps', default=5000, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--data_dir', default='../data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='results', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./runs', type=str)
    parser.add_argument('--interpolation_K', default=10, type=int)
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--interpolation_batch_size', default=5, type=int)
    parser.add_argument('--manual_samples', default=0, type=int)
    parser.add_argument('--fid', action='store_true')
    args = parser.parse_args()

    # config = 'z-{}_batch-{}_num-train-steps-{}'.format(args.latent_dim, args.batch_size, args.num_training_steps)
    config = 'z-{}_gh-{}_dh-{}_batch-{}_num-train-steps-{}'.format(args.latent_dim, args.generator_hidden_dim, args.discriminator_hidden_dim, args.batch_size, args.num_training_steps)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = Dataset(args.batch_size, args.data_dir)
    netG = GAN.get_generator(1, args.latent_dim, args.generator_hidden_dim, device)
    netD = GAN.get_discriminator(1, args.discriminator_hidden_dim, device)
    tb_writer = SummaryWriter(args.log_dir)

    if args.do_train:
        optimG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        optimD = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        trainer = Trainer(device, netG, netD, optimG, optimD, dataset, args.ckpt_dir, tb_writer)
        trainer.train(args.num_training_steps, args.logging_steps, args.saving_steps)

    restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
    netG.restore(restore_ckpt_path)


    if args.interpolate:
        print("Interpolating {} batches with {} steps".format(args.interpolation_batch_size, args.interpolation_K), flush=True)
        with torch.no_grad():
            imgList = []
            for i in range(args.interpolation_batch_size):
                z1 = torch.randn(1, netG.latent_dim, 1, 1, device=device)
                z2 = torch.randn(1, netG.latent_dim, 1, 1, device=device)
                for j in range(args.interpolation_K):
                    z = z1 * (args.interpolation_K - j) / args.interpolation_K + z2 * j / args.interpolation_K
                    img = netG(z).squeeze(0).squeeze(0).cpu()
                    imgList.append(img.unsqueeze(0))
            imgList = torch.stack(imgList, 0)
            imgs = make_grid(imgList, nrow=args.interpolation_K) * 0.5 + 0.5
            tb_writer.add_image('interpolation', imgs)
            save_image(imgs, os.path.join(restore_ckpt_path, "interpolation.png"))
                        
    if args.manual_samples > 0:
        print("Generating {} samples".format(args.manual_samples), flush=True)
        with torch.no_grad():
            latent_vecs = torch.randn(args.manual_samples, netG.latent_dim, 1, 1, device=device)
            imgs = netG(latent_vecs)
            ms_rows = args.manual_samples // 16
            img_grid = make_grid(imgs, nrow=ms_rows) * 0.5 + 0.5
            tb_writer.add_image('manual_samples', img_grid)
            save_image(img_grid, os.path.join(restore_ckpt_path, "manual_samples.png"))
            # save individual imgs
            for i in range(args.manual_samples):
                save_image(imgs[i], os.path.join(restore_ckpt_path, "manual_samples", "ms_{}.png".format(i)))

    num_samples = 3000
    real_imgs = None
    real_dl = iter(dataset.training_loader)
    while real_imgs is None or real_imgs.size(0) < num_samples:
        imgs = next(real_dl)
        if real_imgs is None:
            real_imgs = imgs[0]
        else:
            real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5

    with torch.no_grad():
        samples = None
        while samples is None or samples.size(0) < num_samples:
            imgs = netG.forward(torch.randn(args.batch_size, netG.latent_dim, 1, 1, device=device))
            if samples is None:
                samples = imgs
            else:
                samples = torch.cat((samples, imgs), 0)
    samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    samples = samples.cpu()

    if args.fid:
        fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
        tb_writer.add_scalar('fid', fid)
        print("FID score: {:.3f}".format(fid), flush=True)
        # Create a text file and write the FID score to it.
        with open(os.path.join(args.ckpt_dir, "FIDScore.txt"), 'w') as f:
            f.write(str(fid))