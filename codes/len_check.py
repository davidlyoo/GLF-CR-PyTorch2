import argparse
from sen12mscr_dataset import SEN12MSCR
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=1)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='/data/SEN12MS/SEN12MSCR') 
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    opts = parser.parse_args()

    dataset = SEN12MSCR(opts, opts.input_data_folder)
    dataloader = DataLoader(dataset=dataset, batch_size=opts.batch_sz, shuffle=False)

    print("📦 Number of total samples:", len(dataset))
    print("📊 Number of batches (len(dataloader)):", len(dataloader))

if __name__ == "__main__":
    main()
