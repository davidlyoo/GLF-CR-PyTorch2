import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from metrics import PSNR, SSIM
from sen12mscr_dataset import SEN12MSCR
from net_CR_RDN import RDN_residual_CR

def test(CR_net, opts):    
    data = SEN12MSCR(opts, opts.input_data_folder)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=opts.batch_sz,shuffle=False)

    os.makedirs('./results', exist_ok=True)
    log_file = open('./results/log.txt', 'w')  # 로그 파일 저장

    def log_print(msg):
        print(msg)
        log_file.write(msg + '\n')

    log_print(f"Total samples: {len(data)}")
    log_print(f"Total batches: {len(dataloader)}")

    results_list = []
    total_psnr = 0.0
    total_ssim = 0.0
    iters = 0
    
    with torch.no_grad():
        for inputs in dataloader:
            cloudy_data     = inputs['cloudy_data'].cuda()
            cloudfree_data  = inputs['cloudfree_data'].cuda()
            SAR_data        = inputs['SAR_data'].cuda()
            file_name = inputs['file_name']

            pred = CR_net(cloudy_data, SAR_data)

            psnr = PSNR(pred, cloudfree_data)
            ssim = SSIM(pred, cloudfree_data).item()

            log_print(f"{iters}  PSNR: {psnr:.4f}  SSIM: {ssim:.4f}")
            results_list.append([file_name, float(psnr), float(ssim)])

            total_psnr += psnr
            total_ssim += ssim

            if iters < 10:
                save_image_grid(cloudy_data[0], pred[0], cloudfree_data[0], file_name, iters)

            iters += 1

def save_image_grid(cloudy, pred, gt, file_name, idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Cloudy Input', 'Predicted', 'Ground Truth']
    images = [cloudy, pred, gt]

    for i in range(3):
        img = images[i].detach().cpu().numpy()
        if img.shape[0] > 3:
            img = img[:3]
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        axes[i].imshow(img)
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    os.makedirs('./results/images', exist_ok=True)
    plt.savefig(f'./results/images/{idx}_{file_name}.png')
    plt.close()
    
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='/data/SEN12MS/SEN12MSCR') 
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False) 
    parser.add_argument('--cloud_threshold', type=float, default=0.2) # only useful when is_use_cloudmask=True

    opts = parser.parse_args()

    CR_net = RDN_residual_CR(opts.crop_size).cuda() 
    checkpoint = torch.load('./checkpoints/20_net_CR.pth')
    CR_net.load_state_dict(checkpoint['network'])

    CR_net.eval()
    for _,param in CR_net.named_parameters():
        param.requires_grad = False

    test(CR_net, opts)


if __name__ == "__main__":
    main()
