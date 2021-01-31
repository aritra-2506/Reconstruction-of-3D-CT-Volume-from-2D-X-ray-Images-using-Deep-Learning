import numpy as np
import os
import torch
from generate_drr import do_full_prprocessing
import albumentations as A
from torch.utils.data import DataLoader, Dataset
import ray
import psutil


num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)

# dataset paths

train = "/home/daisylabs/aritra_project/dataset/train"
val = "/home/daisylabs/aritra_project/dataset/val"
app = "/home/daisylabs/aritra_project/dataset/app"


class ImageData(Dataset):
    def __init__(self, data, phase_coeff):
        self.root = data
        self.folder = os.listdir(self.root)
        self.folder.sort()
        self.aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, interpolation=1, border_mode=4, always_apply=False, p=0.3),
            A.RandomCrop(220, 220, always_apply=False, p=1.0),
            A.HorizontalFlip(always_apply=False, p=0.2),
            A.VerticalFlip(always_apply=False, p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, always_apply=False, p=0.5),
            A.RandomBrightness(limit=0.2, always_apply=False, p=0.2),
            A.RandomContrast(limit=0.2, always_apply=False, p=0.2),
            A.MedianBlur(blur_limit=5, always_apply=False, p=0.2),
            A.GaussNoise(var_limit=(10, 50), always_apply=False, p=0.2),
            A.Resize(256, 256),
        ])
        self.phase_coeff = phase_coeff

    def __len__(self):
        return (len(self.folder))

    def __getitem__(self, index):
        patient_list = os.listdir(os.path.join(self.root, self.folder[index]))
        patient_list.sort()

        targets = np.load(os.path.join(self.root, self.folder[index], patient_list[0]))
        targets = targets.astype('float32')

        if (self.phase_coeff == 1):
            targets = np.transpose(targets, (1, 2, 0))

            transformed = self.aug(image=targets, mask=targets)
            targets = transformed['mask']
            targets = (targets - np.min(targets)) * (1.0 / (np.max(targets) - np.min(targets)))
            targets = np.transpose(targets, (2, 0, 1))

            targets_ray = ray.put(targets)

            inputs = ray.get([do_full_prprocessing.remote(targets_ray)])

            inputs = np.asarray(inputs)

            inputs[0][1] = np.rot90(inputs[0][1])
            inputs[0][1] = np.rot90(inputs[0][1])
            inputs[0][1] = np.rot90(inputs[0][1])


            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)

        else:

            inputs = []

            inputs_front = np.load(os.path.join(self.root, self.folder[index], patient_list[1]))
            inputs_lat = np.load(os.path.join(self.root, self.folder[index], patient_list[2]))
            inputs_top = np.load(os.path.join(self.root, self.folder[index], patient_list[3]))
            targets = np.load(os.path.join(self.root, self.folder[index], patient_list[0]))

            inputs_front = inputs_front.astype('float32')
            inputs_lat = inputs_lat.astype('float32')
            inputs_top = inputs_top.astype('float32')
            targets = targets.astype('float32')

            inputs.append(inputs_front)
            inputs.append(inputs_lat)
            inputs.append(inputs_top)

            inputs = np.array(inputs)

            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)


        inputs = inputs.cuda()
        targets = targets.cuda()

        return inputs, targets



def loaders(batch_size, phase):

    if (phase == 0):
        dataset = ImageData(train, 1)
    elif (phase == 1):
        dataset = ImageData(val, 0)
    elif (phase == 2):
        dataset = ImageData(app, 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    return loader

