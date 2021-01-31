import torch
import matplotlib.pyplot as plt
from data_loader import loaders
from network import UNet

def my_app():

    batch_size_app = 1
    loader_ap = loaders(batch_size_app, 2)

    output = UNet()
    output.cuda()

    #output.load_state_dict(torch.load('/home/daisylabs/aritra_project/results/output.pth'))

    output.load_state_dict(torch.load('/home/daisylabs/aritra_project/results/output_best.pth'))

    output.eval()

    with torch.set_grad_enabled(False):
        for u, (inputs, targets) in enumerate(loader_ap):
            if (u == 0):
                inputs = inputs.reshape((batch_size_app, 3, 256, 256))
                targets = targets.reshape((batch_size_app, 256, 256, 256))

                out_1, out_2 = output(inputs)

                out_1 = out_1.reshape((batch_size_app, 256, 256, 256))

                targets = targets.cpu().numpy()
                out_1 = out_1.cpu().numpy()

                tgt_shp = targets.shape[1]

                for slice_number in range(tgt_shp):
                    targets_1 = targets[0][slice_number].reshape((256, 256))
                    out_1_1 = out_1[0][slice_number].reshape((256, 256))

                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.title('Original Slice')
                    plt.imshow(targets_1, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
                    plt.subplot(1, 2, 2)
                    plt.title('Reconstructed Slice')
                    plt.imshow(out_1_1, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

                    plt.savefig('/home/daisylabs/aritra_project/results/slices/%d.png' % (
                    slice_number + 1,))

            else:
                break
