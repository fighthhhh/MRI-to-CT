best_epoch = 0
best_metric = float('inf')
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    train()
    metric = self.loss_G
    if metric < best_metric:
        best_metric = metric
        best_epoch = epoch
        model.save_networks('best')
print('Best epoch:', best_epoch)
print('Best metric:', best_metric)
