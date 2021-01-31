import torch
import gc
import loss_metric

device = torch.device("cuda:1")

def my_eval(output, loader_vl, no_of_batches_1, no_of_epochs, epoch):
    output.eval()

    running_val_loss = 0.0
    running_val_metric = 0.0
    running_val_metric1 = 0.0

    batch_index = 1
    samples = 1

    with torch.set_grad_enabled(False):
        for u, (inputs, targets) in enumerate(
                loader_vl):

            batch_length = len(inputs)

            inputs = inputs.reshape((batch_length, 3, 256, 256))
            targets = targets.reshape((batch_length, 256, 256, 256))

            out_1, out_2 = output(inputs)

            out_1 = out_1.reshape((batch_length, 256, 256, 256))
            out_2 = out_2.reshape((batch_length, 3, 256, 256))

            val_loss_1 = loss_metric.loss1(out_1, targets)
            val_loss_2 = loss_metric.loss2(out_2, inputs)

            val_loss = val_loss_1 + 0.5 * val_loss_2
            running_val_loss = running_val_loss + val_loss.item()

            val_metric = loss_metric.psnr(out_1, targets)
            running_val_metric = running_val_metric + val_metric

            val_metric1 = loss_metric.ssim(out_1, targets)
            running_val_metric1 = running_val_metric1 + val_metric1.item()

            print('batch', batch_index, 'of', no_of_batches_1, 'epoch', epoch + 1, 'of', no_of_epochs, 'samples', '(', samples, '-',
                  samples + batch_length - 1, ')', '-', 'val-loss', ':',
                  "%.3f" % round((val_loss.item()), 3), '-', 'val-PSNR(dB)', ':', "%.3f" % round((val_metric), 3), '-',
                  'val-SSIM', ':', "%.3f" % round((val_metric1.item()), 3))
            batch_index = batch_index + 1
            samples = samples + batch_length

    running_val_loss = running_val_loss / no_of_batches_1
    running_val_metric = running_val_metric / no_of_batches_1
    running_val_metric1 = running_val_metric1 / no_of_batches_1

    del inputs
    del targets
    gc.collect()
    torch.cuda.empty_cache()

    return running_val_loss, running_val_metric, running_val_metric1

