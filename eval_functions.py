import numpy as np
from tqdm import tqdm

from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from glob import glob

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
# Code from: https://github.com/OldaKodym/evaluation_metrics/blob/master/metrics.py

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    footprint = generate_binary_structure(result.ndim, connectivity)
              
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def hd95(result, reference, voxelspacing=None, connectivity=1):
# Code from: https://github.com/OldaKodym/evaluation_metrics/blob/master/metrics.py

    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def dc(input1, input2):
# Code from: https://github.com/OldaKodym/evaluation_metrics/blob/master/metrics.py

    input1 = np.atleast_1d(input1.astype(bool))
    input2 = np.atleast_1d(input2.astype(bool))
    
    intersection = np.count_nonzero(input1 & input2)
    
    size_i1 = np.count_nonzero(input1)
    size_i2 = np.count_nonzero(input2)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def count(model, dataloader, dc, hd95) -> None:
    """
    Function counting both Dice coefficient and Hausdorf distance.

    Args:
        model: network architecture
        dataloader: dataloader to evaluate
    """
    dc_tab = []
    hd_tab = []

    for image, label in tqdm(dataloader):
        # image = image.to(device)
        # label = label.to(device)
        image = image.squeeze(0)
        image = image.permute(0,4,1,2,3)
        prediction = model(image)

        image = image.squeeze(0).squeeze(0)
        label = label.squeeze(0).squeeze(0)
        image = image.detach().to("cpu").numpy()
        label = label.detach().to("cpu").numpy()

        prediction = prediction.squeeze(0).squeeze(0)
        prediction = prediction.detach().to("cpu").numpy()
        prediction[prediction > 0.5] = 1
        prediction[prediction <= 0.5] = 0

        dc_tab.append(dc(image, prediction))
        hd_tab.append(hd95(image, prediction))

        dc_mean = sum(dc_tab)/len(dc_tab)
        hd_mean = sum(hd_tab)/len(hd_tab)

        dc_std = np.std(dc_tab)
        hd_std = np.std(hd_tab)

    print('Dice score: ' + str(dc_mean) + ' +/- ' + str(dc_std))
    print('HD95: ' + str(hd_mean) + ' +/- ' + str(hd_std))


