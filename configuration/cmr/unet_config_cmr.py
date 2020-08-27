from loaders import cmr
loader = cmr

params = {
    'normalise': 'batch',
    'seed': 1,
    'folder': 'experiment_unet_cmr',
    'epochs': 150,
    'batch_size': 8,
    'split': 0,
    'dataset_name': 'cmr',
    'test_dataset': 'cmr',
    'prefix': 'norm',  # Prefix used to load a dataset, e.g. norm_baseline for data_by_dog
    'augment': True,
    'model': 'unet.UNet',
    'executor': 'base_executor.Executor',
    'num_masks': loader.CmrLoader().num_patho_masks + loader.CmrLoader().num_anato_masks,
    'out_channels': loader.CmrLoader().num_patho_masks + loader.CmrLoader().num_anato_masks + 1,
    'outputs':1, # harric added 20191004
    'residual': False,
    'deep_supervision': False,
    'filters': 48,
    'downsample': 3,
    'input_shape': loader.CmrLoader().input_shape,  # harric modified
    'modality': 'LGE',
    'image_downsample': 1,
    'lr': 0.0001,
    'l_mix': 1,
    'decay': 0.,
    'regularizer': 0,
    'ce_weight': 0.3
}


def get(segmentation_option):
    return params
