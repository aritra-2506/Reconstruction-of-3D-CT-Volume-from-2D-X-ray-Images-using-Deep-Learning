import numpy as np # linear algebra
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from configparser import ConfigParser
import warnings
import pylidc as pl
from numba import jit
import ray
from skimage import io
import psutil
from scipy.ndimage import zoom
import cv2

warnings.filterwarnings(action='ignore')

def plot_3d(image, threshold=-300):
	# Position the scan upright,
	# so the head of the patient would be at the top facing the camera
	p = image.transpose(2, 1, 0)

	verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')

	# Fancy indexing: `verts[faces]` to generate a collection of triangles
	mesh = Poly3DCollection(verts[faces], alpha=0.70)
	face_color = [0.45, 0.45, 0.75]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)

	ax.set_xlim(0, p.shape[0])
	ax.set_ylim(0, p.shape[1])
	ax.set_zlim(0, p.shape[2])

	plt.show()

# Load the scans in given folder path
def load_scan(path):
	slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
	slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
	try:
		slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

	for s in slices:
		s.SliceThickness = slice_thickness

	return slices


def convert_dcm_to_npy(images):
	volume = np.stack(
		[
			x.pixel_array * x.RescaleSlope + x.RescaleIntercept
			for x in images
		],
		axis=-1,
	).astype(np.int16)
	return volume


def get_pixels_hu(slices):
	image = np.stack([s.pixel_array for s in slices])
	# Convert to int16 (from sometimes int16),
	# should be possible as values should always be low enough (<32k)
	image = image.astype(np.int16)

	# Set outside-of-scan pixels to 0
	# The intercept is usually -1024, so air is approximately 0
	image[image == -2000] = 0

	# Convert to Hounsfield units (HU)
	for slice_number in range(len(slices)):

		intercept = slices[slice_number].RescaleIntercept
		slope = slices[slice_number].RescaleSlope

		if slope != 1:
			image[slice_number] = slope * image[slice_number].astype(np.float64)
			image[slice_number] = image[slice_number].astype(np.int16)

		image[slice_number] += np.int16(intercept)

	return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
	# Determine current pixel spacing
	spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

	resize_factor = spacing / new_spacing
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = spacing / real_resize_factor

	image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

	return image, new_spacing

@jit(nopython=True, parallel=True)
def generate_drr_from_ct(ct_scan, direction='frontal'):
	input_shape = ct_scan.shape
	if direction == 'lateral':
		ct_scan = np.transpose(ct_scan, axes=(0, 2, 1))
		input_shape = ct_scan.shape
	elif direction == "top":
		ct_scan = np.transpose(ct_scan, axes=(1, 0, 2))
		input_shape = ct_scan.shape

	drr_out = np.zeros((input_shape[0], input_shape[2]), dtype=np.float32)
	for x in range(input_shape[0]):
		for z in range(input_shape[2]):
			u_av = 0.0
			for y in range(input_shape[1]):
				u_av += 0.2 * (ct_scan[x, y, z] + 1000) / (input_shape[1] * 1000)
			drr_out[x, z] = np.exp(0.02 + u_av)
	return drr_out

@ray.remote
def do_full_prprocessing(patients, output_folder, pat_idxs):
	out_meta = []
	for i in pat_idxs:
		scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patients[i]).first()
		dcm_slices = scan.load_all_dicom_images()
		patient_pixels = get_pixels_hu(dcm_slices)
		pix_resampled, spacing = resample(patient_pixels, dcm_slices, [1, 1, 1])
		if not os.path.isdir(os.path.join(output_folder, patients[i])):
			os.makedirs(os.path.join(output_folder, patients[i]))


		drr_front = generate_drr_from_ct(pix_resampled, direction='frontal')
		drr_lat = generate_drr_from_ct(pix_resampled, direction='lateral')
		drr_top = generate_drr_from_ct(pix_resampled, direction='top')

		pix_resampled = np.transpose(pix_resampled, axes=(1, 0, 2))
		org_shape = pix_resampled.shape
		pix_resampled = zoom(pix_resampled, (512 / org_shape[0], 512 / org_shape[1], 512 / org_shape[2]))
		pix_resampled = (pix_resampled - np.min(pix_resampled)) * (1.0 / (np.max(pix_resampled) - np.min(pix_resampled)))
		np.save(os.path.join(output_folder, patients[i], f"{patients[i]}.npy"), pix_resampled)

		drr_front = cv2.resize(drr_front, (512, 512), interpolation=cv2.INTER_LINEAR)
		drr_front = (drr_front - np.min(drr_front)) * (1.0 / (np.max(drr_front) - np.min(drr_front)))
		np.save(os.path.join(output_folder, patients[i], f"{patients[i]}_drrFrontal.npy"), drr_front)

		drr_lat = cv2.resize(drr_lat, (512, 512), interpolation=cv2.INTER_LINEAR)
		drr_lat = (drr_lat - np.min(drr_lat)) * (1.0 / (np.max(drr_lat) - np.min(drr_lat)))
		np.save(os.path.join(output_folder, patients[i], f"{patients[i]}_drrLateral.npy"), drr_lat)

		drr_top = cv2.resize(drr_top, (512, 512), interpolation=cv2.INTER_LINEAR)
		drr_top = (drr_top - np.min(drr_top)) * (1.0 / (np.max(drr_top) - np.min(drr_top)))
		np.save(os.path.join(output_folder, patients[i], f"{patients[i]}_drrTop.npy"), drr_top)

		out_meta.append((i, spacing))
	return out_meta


# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('./lidc.conf')

# Some constants
input_folder = '/home/daisylabs/aritra_project/LIDC-IDRI/'
output_folder = '/home/daisylabs/aritra_project/dataset'
patients = os.listdir(input_folder)
patients.sort()

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)
pat_idxs = np.array_split(np.arange(len(patients)), num_cpus)
patients = ray.put(patients)
output_folder = ray.put(output_folder)
meta_infos = ray.get([do_full_prprocessing.remote(patients, output_folder, pat_idxs[i]) for i in range(num_cpus)])
ray.shutdown()
print(meta_infos)

