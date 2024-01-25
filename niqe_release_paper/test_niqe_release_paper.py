import os
import numpy as np
import cv2
from scipy.io import loadmat
from skimage.util import view_as_blocks
from scipy.special import gamma

def computefeature(structdis):
    feat = []

    [alpha, betal, betar] = estimateaggdparam(structdis.flatten())
    feat = np.concatenate((feat, [alpha, (betal + betar) / 2]))

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]

    for itr_shift in range(4):
        shifted_structdis = np.roll(structdis, shifts[itr_shift], axis=(0, 1))
        pair = structdis.flatten() * shifted_structdis.flatten()
        [alpha, betal, betar] = estimateaggdparam(pair)
        meanparam = (betar - betal) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat = np.concatenate((feat, [alpha, meanparam, betal, betar]))

    return feat

def computemean(patch):
    return np.mean(patch)

def estimateaggdparam(vec):
    gam = np.arange(0.2, 10, 0.001)
    r_gam = ((gamma(2 / gam))**2) / (gamma(1 / gam) * gamma(3 / gam))

    print("vec:", vec)
    print("vec > 0:", vec[vec > 0])

    left_values = vec[vec < 0]
    right_values = vec[vec > 0]

    print("left_values:", left_values)
    print("right_values:", right_values)

    # Handle overflow in standard deviation calculation
    leftstd = np.sqrt(np.nanmean(np.square(np.clip(left_values, -1e100, 1e100)))) if len(left_values) > 0 else np.nan
    rightstd = np.sqrt(np.nanmean(np.square(np.clip(right_values, -1e100, 1e100)))) if len(right_values) > 0 else np.nan

    print("leftstd:", leftstd)
    print("rightstd:", rightstd)

    if np.isnan(leftstd) or np.isnan(rightstd) or np.isinf(leftstd) or np.isinf(rightstd) or rightstd == 0:
        # Handle NaN, infinity, or zero values
        alpha, betal, betar = np.nan, np.nan, np.nan
    else:
        gammahat = leftstd / rightstd if rightstd != 0 else 0
        rhat = (np.nanmean(np.abs(vec)))**2 / np.nanmean((vec)**2)
        rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
        array_position = np.argmin((r_gam - rhatnorm)**2)
        alpha = gam[array_position]

        betal = leftstd * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
        betar = rightstd * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))

    print("alpha:", alpha)
    print("betal:", betal)
    print("betar:", betar)

    return alpha, betal, betar


def estimateaggdparam_old(vec):
    gam = np.arange(0.2, 10, 0.001)
    r_gam = ((gamma(2 / gam))**2) / (gamma(1 / gam) * gamma(3 / gam))

    leftstd = np.sqrt(np.mean((vec[vec < 0])**2))
    # rightstd = np.sqrt(np.mean((vec[vec > 0])**2))
    rightstd = np.sqrt(np.mean((vec[vec > 0]) ** 2)) if np.any(vec > 0) else 0

    print("leftstd:", leftstd)
    print("rightstd:", rightstd)

    # gammahat = leftstd / rightstd
    gammahat = leftstd / rightstd if rightstd != 0 else 0
    rhat = (np.mean(np.abs(vec)))**2 / np.mean((vec)**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)
    alpha = gam[array_position]

    betal = leftstd * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    betar = rightstd * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))

    return alpha, betal, betar

def computequality(im, blocksizerow, blocksizecol, blockrowoverlap, blockcoloverlap, mu_prisparam, cov_prisparam):
    featnum = 18
    scalenum = 2
    all_feat = []  # Store features at each scale

    for itr_scale in range(1, scalenum + 1):
        # Resize the image to ensure it's evenly divisible by your block size
        resize_height = blocksizerow * 2**(itr_scale - 1)
        resize_width = blocksizecol * 2**(itr_scale - 1)
        im_resized = cv2.resize(im, (resize_width, resize_height))

        mu = cv2.filter2D(im_resized, -1, np.ones((7, 7), np.float32) / 36, borderType=cv2.BORDER_REPLICATE)
        mu_sq = mu * mu
        sigma = np.sqrt(np.abs(cv2.filter2D(im_resized * im_resized, -1, np.ones((7, 7), np.float32) / 36, borderType=cv2.BORDER_REPLICATE) - mu_sq))
        structdis = (im_resized - mu) / (sigma + 1)

        block_size = blocksizerow // itr_scale
        block_shape = (block_size, block_size, structdis.shape[2])

        if itr_scale == 1:
            sharpness = view_as_blocks(sigma, (block_size, block_size, structdis.shape[2]))
            sharpness = sharpness.reshape(-1, *block_shape)
            sharpness = np.array([computemean(block) for block in sharpness if block.size > 0])
            sharpness = sharpness.flatten()

        blocks = view_as_blocks(structdis, block_shape)
        feat_scale = np.array([computefeature(block) for block in blocks.reshape(-1, *block_shape) if block.size > 0])
        feat_scale = feat_scale.reshape((featnum, -1))
        feat_scale = feat_scale.T

        all_feat.append(feat_scale)  # Store features for this scale

    # Find the minimum number of rows among all scales
    min_rows = min(feat.shape[0] for feat in all_feat)

    # Concatenate features from all scales along axis 1
    feat_concatenated = np.concatenate([feat[:min_rows] for feat in all_feat], axis=1)

    distparam = feat_concatenated
    #mu_distparam = np.nanmean(distparam, axis=0)
    mu_distparam = np.mean(np.nan_to_num(distparam), axis=0)
    cov_distparam = np.cov(distparam, rowvar=False)

    # Add a small regularization term to the covariance matrix
    reg_term = 1e-5
    cov_combined = (cov_prisparam + cov_distparam) / 2
    #diag_vals = np.diag(cov_combined).copy()
    #diag_vals += reg_term  # Add regularization term to diagonal
    #cov_combined = np.diag(diag_vals)
    np.fill_diagonal(cov_combined, cov_combined.diagonal() + reg_term)  # Add regularization term to diagonal

    #try:
        #1: invcov_param = np.linalg.pinv(cov_combined)
        #2: invcov_param = np.linalg.pinv(cov_combined, rcond=1e-5)
    #except np.linalg.LinAlgError:
        # Handle cases where the matrix is not invertible
        #1: invcov_param = np.linalg.pinv(cov_combined + reg_term * np.eye(cov_combined.shape[0]))
        #2: invcov_param = np.linalg.pinv(cov_combined + reg_term * np.eye(cov_combined.shape[0]), rcond=1e-5)

    try:
        # Check if the covariance matrix is positive definite
        np.linalg.cholesky(cov_combined)

        # If successful, compute the pseudo-inverse
        # invcov_param = np.linalg.pinv(cov_combined, rcond=1e-5)
        invcov_param = np.linalg.pinv(cov_combined, hermitian=True, rcond=1e-5)

        print("Shapes:", (mu_prisparam - mu_distparam).shape, invcov_param.shape, (mu_prisparam - mu_distparam).shape)

    except np.linalg.LinAlgError:
        # Handle cases where the matrix is not invertible
        # invcov_param = np.linalg.pinv(cov_combined + reg_term * np.eye(cov_combined.shape[0]), rcond=1e-5)
        invcov_param = np.linalg.pinv(cov_combined + reg_term * np.eye(cov_combined.shape[0]), hermitian=True, rcond=1e-5)

    # quality = np.sqrt((mu_prisparam - mu_distparam) @ invcov_param @ (mu_prisparam - mu_distparam))
    quality = np.sqrt(((mu_prisparam - mu_distparam).reshape(-1, 1).T @ invcov_param @ (mu_prisparam - mu_distparam).reshape(-1, 1))[0, 0])

    return quality


if __name__ == "__main__":
    # Load Model Parameters
    mat_contents = loadmat(os.path.join('', 'modelparameters.mat'))
    mu_prisparam = mat_contents['mu_prisparam']
    cov_prisparam = mat_contents['cov_prisparam']

    # Test Images
    for i in range(1, 5):
        image = cv2.imread(os.path.join('', f'image{i}.bmp'))
        quality = computequality(image, 96, 96, 0, 0, mu_prisparam, cov_prisparam)
        print(f'Quality for image{i}.bmp:', quality)
