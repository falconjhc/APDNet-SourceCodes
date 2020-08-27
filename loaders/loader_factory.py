from loaders.rohan import RohanLoader # harric added
from loaders.miccai import MiccaiLoader
from loaders.cmr import CmrLoader
from loaders.liverct import LiverCtLoader
from loaders.kits import KitsLoader
from loaders.toy import ToyLoader
from loaders.multimodalcardiac import MultiModalCardiacLoader

def init_loader(dataset):
    """
    Factory method for initialising data loaders by name.
    """
    if dataset == 'rohan':
        return RohanLoader() # harric added
    elif dataset == 'miccai':
        return MiccaiLoader()
    elif dataset == 'cmr':
        return CmrLoader()

    elif dataset == 'liverct':
        return LiverCtLoader()
    elif dataset == 'kits':
        return KitsLoader()
    elif dataset == 'toy':
        return ToyLoader()
    elif dataset == 'multimodalcardiac':
        return MultiModalCardiacLoader()
    return None