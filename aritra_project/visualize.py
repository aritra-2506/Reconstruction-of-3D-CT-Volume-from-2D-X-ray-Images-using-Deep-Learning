import torch
import matplotlib.pyplot as plt

def my_vis(epoch_values, loss_values, val_loss_values, metric_values, val_metric_values, metric1_values, val_metric1_values, output, best_metric_coeff):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Number of epochs')
    ax1.plot(epoch_values, loss_values)
    ax1.plot(epoch_values, val_loss_values)
    ax1.legend(['Train', 'Val'])
    plt.savefig('/home/daisylabs/aritra_project/results/loss.png')
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.set_title('Model PSNR(dB)')
    ax2.set_ylabel('PSNR(dB)')
    ax2.set_xlabel('Number of epochs')
    ax2.plot(epoch_values, metric_values)
    ax2.plot(epoch_values, val_metric_values)
    ax2.legend(['Train', 'Val'])
    plt.savefig('/home/daisylabs/aritra_project/results/PSNR.png')
    plt.show()

    fig3, ax3 = plt.subplots()
    ax3.set_title('Model SSIM(%)')
    ax3.set_ylabel('SSIM(%)')
    ax3.set_xlabel('Number of epochs')
    ax3.plot(epoch_values, metric1_values)
    ax3.plot(epoch_values, val_metric1_values)
    ax3.legend(['Train', 'Val'])
    plt.savefig('/home/daisylabs/aritra_project/results/SSIM.png')
    plt.show()

    op_st_dct = output.state_dict()

    torch.save(op_st_dct, '/home/daisylabs/aritra_project/results/output.pth')

    if(best_metric_coeff==1):
        torch.save(op_st_dct, '/home/daisylabs/aritra_project/results/output_best.pth')
        print('Best Output State Updated')
    else:
        print('Best Output State Retained')

    return


