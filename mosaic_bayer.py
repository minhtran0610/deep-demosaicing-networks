import numpy as np

def mosaic_bayer(rgb, pattern='grbg', noiselevel=0):
    num = np.zeros(len(pattern), dtype=int)
    for i in range(len(pattern)):
        if pattern[i] == 'r':
            num[i] = 0
        elif pattern[i] == 'g':
            num[i] = 1
        elif pattern[i] == 'b':
            num[i] = 2

    mosaic = np.zeros(rgb.shape)
    mask = np.zeros(rgb.shape)
    B = np.zeros(rgb.shape[:2])

    rows1 = np.arange(0, rgb.shape[0], 2)
    rows2 = np.arange(1, rgb.shape[0], 2)
    cols1 = np.arange(0, rgb.shape[1], 2)
    cols2 = np.arange(1, rgb.shape[1], 2)

    B[rows1[:,None], cols1] = rgb[rows1[:,None], cols1, num[0]]
    B[rows1[:,None], cols2] = rgb[rows1[:,None], cols2, num[1]]
    B[rows2[:,None], cols1] = rgb[rows2[:,None], cols1, num[2]]
    B[rows2[:,None], cols2] = rgb[rows2[:,None], cols2, num[3]]

    np.random.seed(0)
    B += noiselevel/255*np.random.randn(*B.shape)

    mask[rows1[:,None], cols1, num[0]] = 1
    mask[rows1[:,None], cols2, num[1]] = 1
    mask[rows2[:,None], cols1, num[2]] = 1
    mask[rows2[:,None], cols2, num[3]] = 1

    mosaic[rows1[:,None], cols1, num[0]] = B[rows1[:,None], cols1]
    mosaic[rows1[:,None], cols2, num[1]] = B[rows1[:,None], cols2]
    mosaic[rows2[:,None], cols1, num[2]] = B[rows2[:,None], cols1]
    mosaic[rows2[:,None], cols2, num[3]] = B[rows2[:,None], cols2]

    return B, mosaic, mask
