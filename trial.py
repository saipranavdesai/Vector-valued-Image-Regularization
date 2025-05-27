import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from Denoise import denoise
from Restore import restore
from removeobject import removemask
from Reconstruct import reconstruct

denoise("a.png",7,75,5)
# restore("bheem.jpg",5,50,0.33,5)
# reconstruct("tree.jpg",10,100,5,21)
# removemask("mask.png","parrot.png",5,50,21)