
import os
import nibabel as nib
import numpy as np
from skimage import transform
from imageio import imread

import utils.data_utils
from loaders.base_loader import Loader

from loaders.data import Data
from parameters import conf
import logging
check_threshold = 5

class CmrLoader(Loader):

    def __init__(self):
        super(CmrLoader, self).__init__()
        self.num_anato_masks = 1
        self.num_patho_masks = 1
        self.num_volumes = 75
        self.input_shape = (192, 192, 1)
        self.data_folder = conf['cmr']
        self.log = logging.getLogger('cmr')

    def splits(self):
        """
        :return: an array of splits into validation, test and train indices
        """

        splits = [
            {'validation': [21,22,23,24,25,26], # --> test on p11
             'test': [21,22,23,24,25,26],
             'training': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
             },

            {'validation': [1, 2, 3, 4, 5, 6],  # --> test on p11
             'test': [1, 2, 3, 4, 5, 6],
             'training': [21, 22, 23, 24, 25, 26, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
             }


        ]

        return splits

    def load_labelled_data(self, split, split_type, modality='LGE',
                           normalise=True, value_crop=True, downsample=1, segmentation_option=-1):
        """
        Load labelled data, and return a Data object. In ACDC there are ES and ED annotations. Preprocessed data
        are saved in .npz files. If they don't exist, load the original images and preprocess.

        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :param downsample:      Downsample data to smaller size. Only used for testing.
        :return:                a Data object
        """
        # if segmentation_option == 0:
        #     input("Segmentation 0")

        if split < 0 or split > 4:
            raise ValueError('Invalid value for split: %d. Allowed values are 0, 1, 2.' % split)
        if split_type not in ['training', 'validation', 'test', 'all']:
            raise ValueError('Invalid value for split_type: %s. Allowed values are training, validation, test, all'
                             % split_type)

        npz_prefix = 'norm_' if normalise else 'unnorm_'

        # If numpy arrays are not saved, load and process raw data
        if not os.path.exists(os.path.join(self.data_folder, npz_prefix + 'cmr_image.npz')):
            if modality == 'LGE':
                value_crop = False
            images, masks_my, masks_ven, masks_mi, patient_index, index, slice = \
                self.load_raw_labelled_data(normalise, value_crop)

            # save numpy arrays
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'cmr_image'), images)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'cmr_myo_mask'), masks_my)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'cmr_ven_mask'), masks_ven)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'cmr_infarct_threshold_%d' % check_threshold), masks_mi)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'cmr_patienet_index'), patient_index)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'cmr_index'),index)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'cmr_slice'), slice)
        # Load data from saved numpy arrays
        else:
            images        = np.load(os.path.join(self.data_folder, npz_prefix + 'cmr_image.npz'))['arr_0']
            masks_my      = np.load(os.path.join(self.data_folder, npz_prefix + 'cmr_myo_mask.npz'))['arr_0']
            masks_ven = np.load(os.path.join(self.data_folder, npz_prefix + 'cmr_ven_mask.npz'))['arr_0']
            masks_mi      = np.load(os.path.join(self.data_folder, npz_prefix + 'cmr_infarct_threshold_%d.npz' % check_threshold))['arr_0']
            patient_index = np.load(os.path.join(self.data_folder, npz_prefix + 'cmr_index.npz'))['arr_0']
            index = np.load(os.path.join(self.data_folder, npz_prefix + 'cmr_index.npz'))['arr_0']
            slice = np.load(os.path.join(self.data_folder, npz_prefix + 'cmr_slice.npz'))['arr_0']

        assert images is not None and masks_my is not None and masks_ven is not None and masks_mi is not None \
               and index is not None, 'Could not find saved data'

        assert images.max() == 1 and images.min() == -1, \
            'Images max=%.3f, min=%.3f' % (images.max(), images.min())

        self.log.debug('Loaded compressed cmr data of shape: ' + str(images.shape) + ' ' + str(index.shape))

        # correct for my segmentation mask (my include mi)
        # anato_masks = np.concatenate([masks_my, masks_ven], axis=-1)
        anato_masks = masks_my
        patho_masks = masks_mi
        # anato_mask_names = ['myocardium', 'ventrincle']
        anato_mask_names = ['myocardium']
        patho_mask_names = ['infarction']
        assert anato_masks.max() == 1 and anato_masks.min() == 0, 'Anatomy Masks max=%.3f, min=%.3f' \
                                                                  % (anato_masks.max(), anato_masks.min())
        assert patho_masks.max() == 1 and patho_masks.min() == 0, 'Pathology Masks max=%.3f, min=%.3f' \
                                                                  % (anato_masks.max(), anato_masks.min())

        scanner = np.array([modality] * index.shape[0])

        # Select images belonging to the volumes of the split_type (training, validation, test)
        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        anato_masks = np.concatenate([anato_masks[index == v] for v in volumes])
        patho_masks = np.concatenate([patho_masks[index == v] for v in volumes])

        assert images.shape[0] == anato_masks.shape[0] == patho_masks.shape[0], "Num of Images inconsistent"

        # create a volume index
        slice = np.concatenate([slice[index == v] for v in volumes])
        index = np.concatenate([index[index == v] for v in volumes])
        scanner = np.array([modality] * index.shape[0])
        assert images.shape[0] == index.shape[0]

        self.log.debug(split_type + ' set: ' + str(images.shape))
        return Data(images, [anato_masks, patho_masks], [anato_mask_names, patho_mask_names], index, slice, scanner,
                    downsample)

    def load_unlabelled_data(self, split, split_type, modality='LGE', normalise=True, value_crop=True):
        """
        Load unlabelled data. In ACDC, this contains images from the cardiac phases between ES and ED.
        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :return:                a Data object
        """
        images, index, slice = self.load_unlabelled_images('cmr', split, split_type, False, normalise, value_crop,modality=modality)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))
        scanner = np.array([modality] * index.shape[0])
        return Data(images, masks, '-1', index, slice, scanner)

    def load_all_data(self, split, split_type, modality='MR', normalise=True, value_crop=True, segmentation_option='-1'):
        """
        Load all images, unlabelled and labelled, meaning all images from all cardiac phases.
        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :return:                a Data object
        """
        images, index, slice = self.load_unlabelled_images('cmr', split, split_type, True, normalise, value_crop,modality=modality)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))
        scanner = np.array([modality] * index.shape[0])
        return Data(images, masks, '-1', index, slice, scanner)

    def load_raw_labelled_data(self, normalise=True, value_crop=True):
        """
        Load labelled data iterating through the ACDC folder structure.
        :param normalise:   normalise data between -1, 1
        :param value_crop:  crop between 5 and 95 percentile
        :return:            a tuple of the image and mask arrays
        """
        self.log.debug('Loading cmr data from original location')
        images, masks_mi, masks_my, masks_ve, patient_index, index, slice = [], [], [], [], [], [], []
        existed_directories = [vol for vol in os.listdir(self.data_folder)
                               if (not vol.startswith('.')) and os.path.isdir(os.path.join(self.data_folder, vol))]
        existed_directories.sort()
        # assert len(existed_directories) == len(self.volumes), 'Incorrect Volume Num !'

        self.volumes = np.unique(self.volumes)
        self.volumes.sort()

        for patient_i in self.volumes:

            patient_images, patient_mi, patient_my, patient_ve = [], [], [], []
            # if not os.path.isdir(os.path.join(self.data_folder,existed_directories[patient_i-1])):
            #     continue
            patient = existed_directories[patient_i-1]

            print('Extracting Labeled Patient: %s @ %d / %d' % (patient, patient_i, len(self.volumes)))


            patient_folder = os.path.join(self.data_folder,patient)
            img_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('00_image')==-1)]
            mi_file_list = [file for file in os.listdir(patient_folder)
                            if (not file.startswith('.')) and (not file.find('infarct_threshold_%d' % check_threshold) == -1)]
            my_file_list = [file for file in os.listdir(patient_folder)
                            if (not file.startswith('.')) and (not file.find('myo_mask') == -1)]
            ve_file_list = [file for file in os.listdir(patient_folder)
                            if (not file.startswith('.')) and (not file.find('ven_mask') == -1)]
            img_file_list.sort()
            mi_file_list.sort()
            my_file_list.sort()
            ve_file_list.sort()
            slices_num = len(img_file_list)

            # for patient index (patient names)
            for ii in range(slices_num):
                patient_index.append(patient)
                index.append(patient_i)

            volume_num = len(img_file_list)
            for v in range(volume_num):
                current_img_name = img_file_list[v]
                current_mi_name = mi_file_list[v]
                current_my_name = my_file_list[v]
                current_ve_name = ve_file_list[v]
                v_id_from_img = current_img_name.split('_')[0]
                v_id_from_mi = current_mi_name.split('_')[0]
                v_id_from_my = current_my_name.split('_')[0]
                v_id_from_ve = current_ve_name.split('_')[0]
                assert v_id_from_img==v_id_from_mi and v_id_from_mi==v_id_from_my and v_id_from_my==v_id_from_ve, \
                    'Mis-Alignment !: Img:%s, MI:%s, MY:%s, VE:%s' % (v_id_from_img, v_id_from_mi, v_id_from_my, v_id_from_ve)
                slice.append(v_id_from_img)

            # for original images
            for org_img_path in img_file_list:
                im = imread(os.path.join(patient_folder,org_img_path))
                # im = im / np.max(im - np.min(im))
                # im = im[:,:,0]
                patient_images.append(np.expand_dims(im,axis=-1))
            patient_images = np.concatenate(patient_images, axis=-1)


            # crop to 5-95 percentile
            if value_crop:
                p5 = np.percentile(patient_images.flatten(), 5)
                p95 = np.percentile(patient_images.flatten(), 95)
                patient_images = np.clip(patient_images, p5, p95)

            # normalise to -1, 1
            if normalise:
                patient_images = utils.data_utils.normalise(patient_images, -1, 1)
            images.append(np.expand_dims(patient_images,axis=-1))

            for mi_seg_path in mi_file_list:
                mi = imread(os.path.join(patient_folder,mi_seg_path))
                if not (len(np.unique(mi)) == 1 and np.unique(mi)[0] == 0):
                    mi = mi / np.max(mi)
                patient_mi.append(np.expand_dims(mi, axis=-1))
            patient_mi = np.concatenate(patient_mi,axis=-1)
            masks_mi.append(np.expand_dims(patient_mi,axis=-1))

            for my_seg_path in my_file_list:
                my = imread(os.path.join(patient_folder,my_seg_path))
                if not (len(np.unique(my)) == 1 and np.unique(my)[0] == 0):
                    my = my / np.max(my)
                patient_my.append(np.expand_dims(my, axis=-1))
            patient_my = np.concatenate(patient_my,axis=-1)
            masks_my.append(np.expand_dims(patient_my, axis=-1))

            for ve_seg_path in ve_file_list:
                ve = imread(os.path.join(patient_folder,ve_seg_path))
                if not (len(np.unique(ve)) == 1 and np.unique(ve)[0] == 0):
                    ve = ve / np.max(ve)
                patient_ve.append(np.expand_dims(ve, axis=-1))
            patient_ve = np.concatenate(patient_ve,axis=-1)
            masks_ve.append(np.expand_dims(patient_ve, axis=-1))




        # move slice axis to the first position
        images = [np.moveaxis(im, 2, 0) for im in images]
        masks_my = [np.moveaxis(m, 2, 0) for m in masks_my]
        masks_mi = [np.moveaxis(m, 2, 0) for m in masks_mi]
        masks_ve = [np.moveaxis(m, 2, 0) for m in masks_ve]

        # crop images and masks to the same pixel dimensions and concatenate all data
        images_cropped, masks_my_cropped = utils.data_utils.crop_same(images, masks_my,
                                                                      (self.input_shape[0], self.input_shape[1]))
        _, masks_mi_cropped = utils.data_utils.crop_same(images, masks_mi,
                                                         (self.input_shape[0], self.input_shape[1]))
        _, masks_ve_cropped = utils.data_utils.crop_same(images, masks_ve,
                                                         (self.input_shape[0], self.input_shape[1]))

        images_cropped = np.concatenate(images_cropped, axis=0)
        masks_my_cropped = np.concatenate(masks_my_cropped, axis=0)
        masks_mi_cropped = np.concatenate(masks_mi_cropped, axis=0)
        masks_ve_cropped = np.concatenate(masks_ve_cropped, axis=0)
        patient_index = np.array(patient_index)
        index = np.array(index)
        slice = np.array(slice)

        return images_cropped, masks_my_cropped, masks_ve_cropped, masks_mi_cropped, patient_index, index, slice

    def resample_raw_image(self, mask_fname, patient_folder, binary=True):
        """
        Load raw data (image/mask) and resample to fixed resolution.
        :param mask_fname:     filename of mask
        :param patient_folder: folder containing patient data
        :param binary:         boolean to define binary masks or not
        :return:               the resampled image
        """
        m_nii_fname = os.path.join(patient_folder, mask_fname)
        new_res = (1.37, 1.37)
        print('Resampling %s at resolution %s to file %s' % (m_nii_fname, str(new_res), new_res))
        im_nii = nib.load(m_nii_fname)
        im_data = im_nii.get_data()
        voxel_size = im_nii.header.get_zooms()

        scale_vector = [voxel_size[i] / new_res[i] for i in range(len(new_res))]
        order = 0 if binary else 1

        result = []
        for i in range(im_data.shape[-1]):
            im = im_data[..., i]
            rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
            result.append(np.expand_dims(rescaled, axis=-1))
        return np.concatenate(result, axis=-1)

    def process_raw_image(self, im_fname, patient_folder, value_crop, normalise):
        """
        Rescale between -1 and 1 and crop extreme values of an image
        :param im_fname:        filename of the image
        :param patient_folder:  folder of patient data
        :param value_crop:      True/False to crop values between 5/95 percentiles
        :param normalise:       True/False normalise images
        :return:                a processed image
        """
        im = self.resample_raw_image(im_fname, patient_folder, binary=False)

        # crop to 5-95 percentile
        if value_crop:
            p5 = np.percentile(im.flatten(), 5)
            p95 = np.percentile(im.flatten(), 95)
            im = np.clip(im, p5, p95)

        # normalise to -1, 1
        if normalise:
            im = utils.data_utils.normalise(im, -1, 1)

        return im

    def load_raw_unlabelled_data(self, include_labelled=True, normalise=True, value_crop=True, modality='LGE'):
        """
        Load unlabelled data iterating through the ACDC folder structure.
        :param include_labelled:    include images from ES, ED phases that are labelled. Can be True/False
        :param normalise:           normalise data between -1, 1
        :param value_crop:          crop between 5 and 95 percentile
        :return:                    an image array
        """
        self.log.debug('Loading unlabelled cmr data from original location')
        images, patient_index, index, slice = [], [], [], []
        existed_directories = [vol for vol in os.listdir(self.data_folder)
                               if (not vol.startswith('.')) and os.path.isdir(os.path.join(self.data_folder,vol))]
        existed_directories.sort()
        # assert len(existed_directories) == len(self.volumes), 'Incorrect Volume Num !'

        self.volumes = np.unique(self.volumes)
        self.volumes.sort()

        for patient_i in self.volumes:
            patient_images = []
            # if not os.path.isdir(os.path.join(self.data_folder,existed_directories[patient_i-1])):
            #     continue
            patient = existed_directories[patient_i-1]
            print('Extracting UnLabeled Patient: %s @ %d / %d' % (patient, patient_i, len(self.volumes)))

            patient_folder = os.path.join(self.data_folder, patient)
            img_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('00_image') == -1)]
            img_file_list.sort()
            slices_num = len(img_file_list)
            for v in range(slices_num):
                current_img_name = img_file_list[v]
                v_id_from_img = current_img_name.split('_')[0]
                slice.append(v_id_from_img)

            # for patient index (patient names)
            for ii in range(slices_num):
                patient_index.append(patient)
                index.append(patient_i)

            # for original images
            for org_img_path in img_file_list:
                im = imread(os.path.join(patient_folder, org_img_path))
                # im = im / np.max(im - np.min(im))
                # im = im[:, :, 0]
                patient_images.append(np.expand_dims(im, axis=-1))
            patient_images = np.concatenate(patient_images, axis=-1)

            # crop to 5-95 percentile
            if value_crop:
                p5 = np.percentile(patient_images.flatten(), 5)
                p95 = np.percentile(patient_images.flatten(), 95)
                patient_images = np.clip(patient_images, p5, p95)

            # normalise to -1, 1
            if normalise:
                patient_images = utils.data_utils.normalise(patient_images, -1, 1)
            images.append(np.expand_dims(patient_images, axis=-1))


        images = [np.moveaxis(im, 2, 0) for im in images]
        zeros = [np.zeros(im.shape) for im in images]
        images_cropped, _ = utils.data_utils.crop_same(images, zeros,
                                                       (self.input_shape[0], self.input_shape[1]))
        images_cropped = np.concatenate(images_cropped, axis=0)[..., 0]
        index = np.array(index)
        slice = np.array(slice)

        return images_cropped, patient_index, index, slice

    def load_unlabelled_images(self, dataset, split, split_type, include_labelled, normalise, value_crop, modality):
        """
        Load only images.
        :param dataset:
        :param split:
        :param split_type:
        :param include_labelled:
        :param normalise:
        :param value_crop:
        :return:
        """
        npz_prefix_type = 'ul_' if not include_labelled else 'all_'
        npz_prefix = npz_prefix_type + 'norm_' if normalise else npz_prefix_type + 'unnorm_'

        # Load saved numpy array
        if os.path.exists(os.path.join(self.data_folder, npz_prefix + 'cmr_image.npz')):
            images = \
                np.load(os.path.join(self.data_folder,
                                     npz_prefix + 'cmr_image.npz'))['arr_0']
            index  = \
                np.load(os.path.join(self.data_folder,
                                     npz_prefix + 'cmr_index.npz'))['arr_0']
            patient_index = \
                np.load(os.path.join(self.data_folder,
                                     npz_prefix + 'cmr_patient_index.npz'))['arr_0']
            slice = \
                np.load(os.path.join(self.data_folder,
                                     npz_prefix + 'cmr_patient_slice.npz'))['arr_0']
            self.log.debug('Loaded compressed ' + dataset + ' unlabelled data of shape ' + str(images.shape))
        # Load from source
        else:
            if modality == 'LGE':
                value_crop = False
            images, patient_index, index, slice = \
                self.load_raw_unlabelled_data(include_labelled, normalise, value_crop, modality=modality)
            images = np.expand_dims(images, axis=3)
            np.savez_compressed(os.path.join(self.data_folder,
                                             npz_prefix + 'cmr_image'), images)
            np.savez_compressed(os.path.join(self.data_folder,
                                             npz_prefix + 'cmr_index'), index)
            np.savez_compressed(os.path.join(self.data_folder,
                                             npz_prefix + 'cmr_patient_index'), patient_index)
            np.savez_compressed(os.path.join(self.data_folder,
                                             npz_prefix + 'cmr_patient_slice'), slice)
        assert split_type in ['training', 'validation', 'test', 'all'], 'Unknown split_type: ' + split_type

        if split_type == 'all':
            return images, index

        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        slice = np.concatenate([slice[index == v] for v in volumes])
        index  = np.concatenate([index[index==v] for v in volumes])
        return images, index, slice