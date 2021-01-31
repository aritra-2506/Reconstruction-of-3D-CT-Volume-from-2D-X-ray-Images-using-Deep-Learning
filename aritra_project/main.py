import torch.optim as optim
from network import UNet
from data_loader import loaders
from train import my_train
from eval import my_eval
from visualize import my_vis
from app import my_app
import numpy as np
import ray

#data loading
batch_size = 2
loader_tr = loaders(batch_size, 0)
loader_vl = loaders(batch_size, 1)

#networks

output = UNet()

output.cuda()

#optimizer

optimizer = optim.Adam(output.parameters(), lr=.00003, weight_decay=1e-4)

#training

metric_values, metric1_values, val_metric_values, val_metric1_values, epoch_values, loss_values, val_loss_values  = ([] for i in range(7))

no_of_epochs = 1000
no_of_batches = len(loader_tr)
no_of_batches_1 = len(loader_vl)
best_metric = 0

for epoch in range(no_of_epochs):
    epoch_loss, epoch_acc, epoch_acc1 = my_train(output, optimizer, loader_tr, no_of_batches,
                                                 no_of_epochs, epoch)

    running_val_loss, running_val_metric, running_val_metric1 = my_eval(output, loader_vl,
                                                                        no_of_batches_1, no_of_epochs, epoch)

    print('epoch', epoch + 1, 'of', no_of_epochs, '-', 'train loss', ':',
          "%.3f" % round((epoch_loss), 3), '-', 'train PSNR(dB)', ':', "%.3f" % round((epoch_acc), 3), '-',
          'train SSIM', ':',
          "%.3f" % round((epoch_acc1), 3), '-', 'val loss', ':', "%.3f" % round((running_val_loss), 3), '-',
          'val PSNR(dB)', ':',
          "%.3f" % round((running_val_metric), 3), '-', 'val SSIM', ':',
          "%.3f" % round((running_val_metric1), 3))

    metric_values.append(round(epoch_acc, 3))
    val_metric_values.append(round(running_val_metric, 3))

    loss_values.append(round(epoch_loss, 3))
    val_loss_values.append(round(running_val_loss, 3))

    metric1_values.append(round(epoch_acc1, 3))
    val_metric1_values.append(round(running_val_metric1, 3))

    current_metric = round(running_val_metric1, 3)

    if(current_metric>best_metric):
        best_metric_coeff = 1
        best_metric = current_metric
    else:
        best_metric_coeff = 0

    epoch_values.append(epoch + 1)

    my_vis(epoch_values, loss_values, val_loss_values, metric_values, val_metric_values, metric1_values,
           val_metric1_values, output, best_metric_coeff)

    vmv = np.amax(np.asarray(val_metric_values))
    vm1v = np.amax(np.asarray(val_metric1_values))
    vlv = np.amin(np.asarray(val_loss_values))

    print('Maximum Validation PSNR(dB)', ':', "%.3f" % vmv)
    print('Maximum Validation SSIM', ':', "%.3f" % vm1v)
    print('Minimum Validation Loss', ':', "%.3f" % vlv)

    np.save('/home/daisylabs/aritra_project/results/val_psnr_values.npy', val_metric_values)
    np.save('/home/daisylabs/aritra_project/results/val_ssim_values.npy', val_metric1_values)
    np.save('/home/daisylabs/aritra_project/results/val_loss_values.npy', val_loss_values)

    np.save('/home/daisylabs/aritra_project/results/psnr_values.npy', metric_values)
    np.save('/home/daisylabs/aritra_project/results/ssim_values.npy', metric1_values)
    np.save('/home/daisylabs/aritra_project/results/loss_values.npy', loss_values)


ray.shutdow()
print('Finished Training')

#app

my_app()
