import numpy as np
from scipy.ndimage import rotate, shift
from skimage.transform import resize

from slmsuite.holography.algorithms import FeedbackHologram
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.toolbox.phase import blaze

from functions import build_weighting_mask

def derive_get_image_func_from_feedback_hologram(cfg, fs: FourierSLM, SLM_pattern, blaze_vector):
    hologram = FeedbackHologram(shape=fs.slm.shape, cameraslm=fs)
    blaze_phase = blaze(grid=fs.slm, vector=blaze_vector)
    hologram.reset_phase(SLM_pattern + blaze_phase)
    center_shift_vector = fs.kxyslm_to_ijcam(blaze_vector)
    M_matrix = fs.calibrations["fourier"]['M']
    rotation_angle = np.arctan2(M_matrix[1, 0], M_matrix[0, 0])

    # we only want the image inside the mask
    Wcg_np = build_weighting_mask(cfg)
    Wcg_rot = rotate(
        Wcg_np,
        rotation_angle,
        reshape=False,
        order=0,
    )
    Wcg = shift(
        Wcg_rot,
        shift=center_shift_vector,
        order=0,
    )

    whole_image = np.array(fs.cam.get_image())
    resized_image = resize(whole_image, Wcg.shape, anti_aliasing=True)
    image = resized_image * Wcg

    def get_image_func():
        return image
    
    return get_image_func