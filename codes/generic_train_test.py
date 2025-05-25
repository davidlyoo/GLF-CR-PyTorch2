from tqdm import tqdm

class Generic_train_test():
    def __init__(self, model, opts, dataloader):
        self.model = model
        self.opts = opts
        self.dataloader = dataloader

    def decode_input(self, data):
        raise NotImplementedError()

    def train(self):
        total_steps = 0
        print()
        print('#training images ', len(self.dataloader) * self.opts.batch_sz)

        for epoch in range(self.opts.max_epochs):
            log_loss = 0
            tqdm_bar = tqdm(self.dataloader, dynamic_ncols=True)

            for step, data in enumerate(tqdm_bar):
                total_steps += 1
                _input = self.decode_input(data)

                self.model.set_input(_input)
                batch_loss = self.model.optimize_parameters()
                log_loss += batch_loss
                
                tqdm_bar.set_description(f"[Epoch {epoch}/{self.opts.max_epochs} | Total Step {total_steps}]")
                
                if total_steps % self.opts.log_freq == 0:
                    info = self.model.get_current_scalars()
                    psnr = info.get('PSNR_train', 0)
                    avg_loss = log_loss / self.opts.log_freq
                    tqdm_bar.set_postfix({
                        "Loss": f"{avg_loss:.4f}",
                        "PSNR": f"{psnr:.2f}"
                    })
                    log_loss = 0

            if epoch % self.opts.save_freq == 0:
                self.model.save_checkpoint(epoch)

            if epoch > self.opts.lr_start_epoch_decay - self.opts.lr_step:
                self.model.update_lr()
