# Reconstruction-of-3D-CT-Volume-from-2D-X-ray-Images-using-Deep-Learning
The network inputs a 2D X-ray/DRR Image from 1/2/3 different views (Frontal/Frontal+Lateral/Frontal+Lateral+Top) and outputs a 3D CT Volume.

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
Decomposition Loss, Reconstruction Loss, Latent Space Loss. They are comprised of weighted combination of first and second norm between output and label pixel values. SSIM (primarily) and PSNR are used as accuracy metrices. 
The maximum validation accuracy (SSIM) that was recorded was 80.2%. 
Best results have been stored inside ‘results’ folder in 6 different stages. Every stage incorporates the best combination from the last stage, while testing a different parameter in its own stage.
 1. Stage 1 - Baseline (Accuracy (SSIM) = 63.5%) : It contains front-view DRR as input and top-view CT as label, both pre-processed by MeVisLab. 
2. Stage 2 - Dataset Preprocessing (Python) (Accuracy = 65%) : It contains same input-label combination as stage 1, except that it has been entirely pre-processed by Python script.
3. Stage 3 - CT-DRR Combination (Frontal) (Accuracy = 72.5%) : It contains front-view DRR and front-view CT as input-label combination.
4. Stage 4 - Loss (Decomposition+Reconstruction) (Accuracy = 72.9%) : It incorporates the reconstruction loss apart from the best combination from last stage, which had only decomposition loss.
5. Stage 5 - Dimension (512 pixels) (Accuracy = 80.2%) : Dimension of input is set at 512_512 pixels and that of CT at 512_512_512 pixels.
6. Stage 6 - Viewpoint (Frontal+Lateral) (Accuracy = 80.2%) : Here apart from front-view DRR, also lateral-view DRR is fed as input.
The final setup uses a 2D U-Net, Adam optimizer, Decomposition + Reconstruction Loss, Frontal+Lateral DRR as input, front-view CT as output, both of dimension 512 pixels and pre-processed by Python script and SSIM and PSNR as metrics. 

PSNR for first stage is 16.339 dB and that for last stage is 22.714 dB. Absolute Loss values are 150214.282 and 86451.029 for first and last stages respectively. 

Apart from the ‘results’ folder, there is ‘Miscellaneous Results’ folder which contains other expeirments which have been conducted. They are:
	1.	Generation of 2D Front-view DRR from 3D front-view CT (Accuracy = 98.1%)
	2.	Reconstruction of 3D Front-view CT from itself (Accuracy = 91.9%)
	3.	Reconstruction of 3D Lateral-view DRR from 3D front+lateral+top-view CT (Accuracy = 72.7%)
	4.	Reconstruction of 3D Top-view DRR from 3D front+lateral+top-view CT (Accuracy = 74%)
	5.      A comparison among original, noisy input and denoised output DRR.
Apart from that, there is a folder of Data Augmentation, which shows augmentation impact on both input and labels for Shift, Scale, Rotate (shift = 0.2, scale = 0.2, rotate = 45 degrees) and Elastic Transform (alpha = 1, beta = 50).

complete_workflow.png, dataset_pre-processing.png, network_architecture.png and network_diagram and depict the complete workflow, data pre-processing method, complete network architecture as functional blocks and actual netork diagram respectively.


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







