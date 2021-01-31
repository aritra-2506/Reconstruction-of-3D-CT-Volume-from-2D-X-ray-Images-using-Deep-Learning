import torch
import gc
import loss_metric

def my_train(output, optimizer, loader_tr, no_of_batches, no_of_epochs, epoch):
    output.train()

    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_acc1 = 0.0

    batch_index = 1
    samples = 1

    for u, (inputs, targets) in enumerate(loader_tr):
        optimizer.zero_grad()

        batch_length = len(inputs)

        inputs = inputs.reshape((batch_length, 3, 256, 256))
        targets = targets.reshape((batch_length, 256, 256, 256))

        out_1, out_2 = output(inputs)

        out_1 = out_1.reshape((batch_length, 256, 256, 256))
        out_2 = out_2.reshape((batch_length, 3, 256, 256))


        loss_1 = loss_metric.loss1(out_1, targets)
        loss_2 = loss_metric.loss2(out_2, inputs)

        loss = loss_1 + 0.5 * loss_2

        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss = epoch_loss + loss.item()

        metric = loss_metric.psnr(out_1, targets)
        epoch_acc = epoch_acc + metric

        metric1 = loss_metric.ssim(out_1, targets)
        epoch_acc1 = epoch_acc1 + metric1.item()

        print('batch', batch_index, 'of', no_of_batches, 'epoch', epoch + 1, 'of', no_of_epochs, 'samples', '(', samples, '-',
              samples + batch_length - 1, ')', '-', 'loss', ':',
              "%.3f" % round((loss.item()), 3), '-', 'PSNR(dB)', ':', "%.3f" % round((metric), 3), '-',
              'SSIM', ':', "%.3f" % round((metric1.item()), 3))

        batch_index = batch_index + 1
        samples = samples + batch_length

    epoch_loss = epoch_loss / no_of_batches
    epoch_acc = epoch_acc / no_of_batches
    epoch_acc1 = epoch_acc1 / no_of_batches

    del inputs
    del targets
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_loss, epoch_acc, epoch_acc1
