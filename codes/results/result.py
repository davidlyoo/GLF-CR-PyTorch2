def parse_log_and_print_average(log_path='./log.txt'):
    psnr_list = []
    ssim_list = []

    with open(log_path, 'r') as f:
        for line in f:
            if 'PSNR:' in line and 'SSIM:' in line:
                parts = line.strip().split()
                try:
                    psnr = float(parts[2])
                    ssim = float(parts[4])
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                except:
                    continue  # skip malformed line

    if len(psnr_list) == 0:
        print("No valid PSNR/SSIM entries found in the log.")
        return

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print(f"Parsed {len(psnr_list)} entries")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    parse_log_and_print_average()
    


# Parsed 122218 entries
# Average PSNR: 31.7006
# Average SSIM: 0.8935