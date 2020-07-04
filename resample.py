import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
import math
from scipy.ndimage.interpolation import rotate
import skimage
from scipy.ndimage import zoom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import datetime, time

PathDicom_W = "/content/drive/My Drive/NewDataset/resave"
# NewPathDicom_W="C:/Users/Aritra Mazumdar/Downloads/ISIC/data_folder/backup"

# PathDicom_W = "C:/Users/Aritra Mazumdar/Downloads/ISIC/data_folder/new_save"


ds2 = pydicom.read_file("/content/drive/My Drive/NewDataset/resave/2.dcm")
ds1 = pydicom.read_file("/content/drive/My Drive/NewDataset/resave/1.dcm")
sliceThickness = ds1.SliceLocation - ds2.SliceLocation
slice1Location = ds1.SliceLocation

lstFilesDCM_W = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom_W):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_W.append(os.path.join(dirName, filename))

images = []

for filenameDCM in lstFilesDCM_W:
    # read the file
    ds = pydicom.read_file(filenameDCM)
    # print(length_of_pixel_array)
    x = ds.pixel_array
    # x=cv2.resize(ds.pixel_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # store the raw image data
    images.append(x)

images = np.asarray(images)

dt = datetime.datetime.now()
siuid=pydicom.uid.generate_uid()


def write_dicom(image2d, i, slicesLocation, dt, siuid):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.FileMetaInformationGroupLength = 150
    meta.ImplementationClassUID = "1.3.6.1.4.1.22213.1.143"

    meta.ImplementationVersionName = '0.5'
    meta.SourceApplicationEntityTitle = 'POSDA'

    filename_little_endian = ("/content/drive/My Drive/NewDataset/dude/%d.dcm" % (i + 1,))
    ds = FileDataset(filename_little_endian, {},
                     file_meta=meta, preamble=b"\0" * 128)

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    ds.PatientName = ''
    ds.PatientID = "LIDC-IDRI-0004"

    ds.StudyDate='20000101'
    ds.SeriesDate='20000101'
    ds.AcquisitionDate='20000101'
    ds.ContentDate='20000101'
    ds.OverlayDate='20000101'
    ds.CurveDate='20000101'
    ds.AcquisitionDateTime='20000101'
    ds.StudyTime=''
    ds.AcquisitioTime=''
    ds.ContentTime=''
    ds.AccessionNumber=''
    ds.Manufacturer='GE MEDICAL SYSTEMS'
    ds.ReferringPhysicianName =''
    ds.ManufacturerModelName ='LightSpeed16'
    ds.PatientBirthDate = ''
    ds.PatientSex=''
    ds.PatientAge=''
    ds.LastMenstrualDate ='20000101'
    ds.DeidentificationMethod  = 'DCM:113100/113105/113107/113108/113109/113111'
    ds.PrivateCreator = 'CTP'
    ds.Privatetagdata = 'LIDC-IDRI'
    ds.Privatetagdata = '62796001'
    ds.ContrastBolusAgent = 'OMNI'
    ds.SoftwareVersions = '06MW03.5'
    ds.DistanceSourcetoDetector = "949.075012"
    ds.DistanceSourcetoPatient= "541.0"
    ds.GantryDetectorTilt = "0.0"
    ds.TableHeight= "174.699997"
    ds.ExposureTime = "690"
    ds.XRayTubeCurrent = "440"
    ds.Exposure="14"
    ds.GeneratorPower="52800"
    ds.FocalSpot ="1.2"
    ds.StudyID=''
    ds.SeriesNumber="3000534"
    ds.PositionReferenceIndicator ='SN'
    ds.PixelPaddingValue = -2000
    ds.ContentCreatorName =''
    ds.StorageMediaFilesetUID=pydicom.uid.generate_uid()
    ds.PersonName = 'Removed by CTP'
    ds.UID=pydicom.uid.generate_uid()
    ds.VerifyingObserverName =  'Removed by CTP'
    ds.AdmittingDate   =     '20000101'
    ds.ScheduledProcedureStepStartDateDA='20000101'
    ds.ScheduledProcedureStepEndDateDA='20000101'
    ds.PerformedProcedureStepStartDateDA='20000101'
    ds.PlacerOrderNumberImagingServiLO= ''
    ds.FilterOrderNumberImagingServiLO= ''







    ds.Modality = "CT"
    ds.SeriesInstanceUID = siuid
    ds.SOPInstanceUID= pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    ds.ReferencedSOPInstanceUID = pydicom.uid.generate_uid()

    # dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15

    ds.ImagesInAcquisition = "1"

    ds.Rows = image2d.shape[0]
    ds.Columns = image2d.shape[1]
    ds.InstanceNumber = i + 1

    # sliceLocation=slice1Location-(sliceThickness*i)
    # print(slicesLocation)
    sliceLocation = float(slicesLocation)
    ds.sliceLocation = sliceLocation
    # print(ds.sliceLocation)

    ds.PixelSpacing = r"0.822266\0.822266"
    print(ds.sliceLocation)
    a = float(-234.800003)
    # a=a.replace(' ','').replace(',','.').replace("−", "-")
    b = float(-170.500000)
    # b=b.replace(' ','').replace(',','.').replace("−", "-")
    print(a)
    ds.ImagePositionPatient = [a, b, sliceLocation]
    # ds.ImagePositionPatient = r"a\b\sliceLocation"

    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.RescaleIntercept = "-1024"
    ds.RescaleSlope = "1"
    ds.ScanOptions = 'HELICAL MODE'

    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1
    ds.NumberofSlices = 150

    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.KVP = 120
    ds.BodyPartExamined = 'CHEST'
    ds.DataCollectionDiameter = "500"
    ds.ReconstructionDiameter = "421"
    ds.RotationDirection = 'CW'
    ds.PatientPosition = 'FFS'
    ds.ConvolutionKernel = 'STANDARD'
    ds.LongitudinalTemporalInformation = 'MODIFIED'
    ds.WindowCenter = '40'
    ds.WindowWidth = '400'

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    print("Setting pixel data...")
    ds.PixelData = image2d.tobytes()
    ds.save_as(filename_little_endian)

    return


z = images.shape[0]
y = 150 / z

images_new = zoom(images, (y, 1, 1))

j = images_new.shape[0]




for i in range(0, j):
    a = images_new[i]
    slicesLocation=float(slice1Location-(sliceThickness*i))
    #print(slicesLocation)
    ds = write_dicom(a, i, slicesLocation, dt, siuid)
