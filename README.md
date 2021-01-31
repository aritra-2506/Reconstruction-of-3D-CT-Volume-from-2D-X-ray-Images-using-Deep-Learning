# Reconstruction-of-3D-CT-Volume-from-2D-X-ray-Images
The network inputs a 2D X-ray Image from 3 different views (Front, Lateral and Top) and outputs a 3D CT Volume.

LIDC Dataset was used in this project. It has 1012 entries of patients. But only 172 suitable entries were selected.

Each entry of patient has unequal number of CT scan slices (300 or 400 or 550) each of dimension (512_512), which constitue the CT volume altogether. Dataset preprocessing needs to be 
done to resample it to 256 number of slices each of dimension (256_256). This CT volume (256_256_256) is the target (/output) of the network. DRR was generated for front, top
and lateral views for each of these CT volumes and these 3 views form the input to the network. Output dimension for a batch size of 2, is [2,256,256,256] and input is [2, 3, 256, 256].

Dataset preprocessing has be done in two different ways:
1. Using MeVisLab
2. Using Python script

For MeVisLab, the workflow of modules responsible for resampling is present in resampling.png and the one responsible for generation of DRRs in drrs.png inside 'mevislab_setup' folder. It stores the files in .dcm format.

For Python script, resampling and DRRs generation are both done using data_generation.py. It stores the files in .npy format.
In the final setup, in the training phase, data augmentation is done to targets (resampled CT volume) and inputs (DRRs) are generated on the fly (generate_drr.py inside aritra_project) with augmented data, whereas in validation phase,
both inputs and targets are generated beforehand statically (data_generation.py), as there is no augmentation, all by using Python scripts.

Besides that, the folder 'aritra_project' has the code which does the the reconstruction task.

Different kinds of image reconstruction networks was used: 2D UNet, 3D UNet, 2D - 3D UNet, UNet weighted with ResNet. Different hyperparameters are tried. Different loss functions have been used:
Decomposition Loss, Reconstruction Loss, Latent Space Loss. SSIM and PSNR are used as accuracy metrices. Best ones have been retained and stored inside 'results' folder.

The final setup uses a 2D UNet, Adam optimizer, Decomposition + Reconstruction Loss and SSIM and PSNR as metrics. SSIM.png, PSNR.png and loss.png are stored in slices inside 'results' folder.

The maximum validation accuracy (SSIM) that was recorded was 72%. Slices 133.png, 166.png and 172.png which are the corrsponding slices out of 256 and reference DRRs, Frontal, Lateral and Top, as DRRs.png, are stored in results folder.





