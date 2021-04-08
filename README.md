# Reconstruction-of-3D-CT-Volume-from-2D-X-ray-Images
The network inputs a 2D X-ray Image from 3 different views (Front, Lateral and Top) and outputs a 3D CT Volume.

LIDC Dataset was used in this project. It has 1012 entries of patients. But only 172 suitable entries were selected.

Each entry of patient has unequal number of CT scan slices (300 or 400 or 550) each of dimension (512_512), which constitue the CT volume altogether. Dataset preprocessing needs to be 
done to resample it to 256 number of slices each of dimension (256_256). This CT volume (256_256_256) is the target (/output) of the network. DRR was generated for front, top
and lateral views for each of these CT volumes and these 3 views form the input to the network. Output dimension for a batch size of 2, is [2, 256, 256, 256] and input is [2, 3, 256, 256].

Dataset preprocessing has be done in two different ways:
1. Using MeVisLab
2. Using Python script

For MeVisLab, the workflow of modules responsible for resampling is present in resampling.png and the one responsible for generation of DRRs in drrs.png inside 'mevislab_setup' folder. It stores the files in .dcm format.

For Python script, resampling and DRRs generation are both done using data_generation.py. It stores the files in .npy format.
In the final setup, in the training phase, data augmentation is done to targets (resampled CT volume) and inputs (DRRs) are generated on the fly (generate_drr.py inside aritra_project) with augmented data, whereas in validation phase,
both inputs and targets are generated beforehand statically (data_generation.py), as there is no augmentation, all by using Python scripts.

Besides that, the folder 'aritra_project' has the code which does the the reconstruction task.

Different kinds of image reconstruction networks was used: 2D U-Net, 3D U-Net, 2D - 3D U-Net, U-Net weighted with ResNet. Different hyperparameters are tried. Different loss functions have been used:
Decomposition Loss, Reconstruction Loss, Latent Space Loss. SSIM and PSNR are used as accuracy metrices. Best ones have been retained and stored inside 'results' folder.

The final setup uses a 2D U-Net, Adam optimizer, Decomposition + Reconstruction Loss and SSIM and PSNR as metrics. SSIM.png, PSNR.png and loss.png are stored in slices inside 'results' folder.

The maximum validation accuracy (SSIM) that was recorded was 72%. Slices 133.png, 166.png and 172.png which are the corrsponding slices out of 256 and reference DRRs, Frontal, Lateral and Top, as DRRs.png inside Reference DRR, are stored in results folder.

final_network.png and dataset_preprocessing.png depict the version of the Network and the complete Data Preprocessing method used for this project.


Steps:

1. Clone the repository. 
2. Download the LIDC Dataset from the internet.
3. Use the data_generation.py file for inputs (DRRs) generation and labels (CT volume) resampling and generation. Before that, change the path to input_folder of LIDC dataset and also to the output_folder, where you want to save the dataset. Alternately, you can use the MeVisLab setup mentioned above for both resampling and data generation. A batch size of 2 would be an input torch tensor of shape [2, 3, 256, 256] (as there are 3 different views of DRR for a single patient and there are 2 patients) and output from the network would be a tensor of shape [2, 256, 256, 256] as it is a volume of are 256 slices (z), each of shape [256 (x) by 256 (y)] each.
4. Save the dataset within the 'aritra_project' folder as dataset and select 'aritra_project' folder as working directory.
5. Select the suitable number of patients from the whole dataset.
6. Change the path to dataset in data_loader.py, the path to saving results in visualize.py, the path to loading state dictionaries and saving the application output in app.py.
7. Install all the necessary libraries present in the code.
8. Run main.py file.
9. SSIM, PSNR and loss and all other essentials will be stored inside 'results' folder inside 'aritra_project' folder. Application output will be stored inside 'slices' folder inside 'results' folder.





