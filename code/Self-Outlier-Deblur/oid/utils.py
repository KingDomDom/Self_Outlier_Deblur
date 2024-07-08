import math
import numpy as np
from scipy.ndimage import shift
import torch


def bwconncomp_2d(bw):
    PixelIdxList = []
    mask = np.zeros(bw.shape, dtype=np.bool8)

    while np.any(bw != mask):
        BW = bw != mask
        r0 = -1
        c0 = -1
        for r in range(BW.shape[0]):
            for c in range(BW.shape[1]):
                if BW[r, c] == True:
                    r0, c0 = r, c
                    break
            if r0 != -1:
                break

        idxlist = [(r0, c0)]

        mask[r0, c0] = True
        k = 0
        while k < len(idxlist):
            r, c = idxlist[k]
            if r - 1 >= 0:
                if c - 1 >= 0 and BW[r - 1, c - 1] == True and (r - 1, c - 1) not in idxlist:
                    idxlist.append((r - 1, c - 1))
                    mask[r - 1, c - 1] = True
                if c + 1 < BW.shape[1] and BW[r - 1, c + 1] == True and (r - 1, c + 1) not in idxlist:
                    idxlist.append((r - 1, c + 1))
                    mask[r - 1, c + 1] = True
            if r + 1 < BW.shape[0]:
                if c - 1 >= 0 and BW[r + 1, c-1] == True and (r + 1, c - 1) not in idxlist:
                    idxlist.append((r + 1, c - 1))
                    mask[r + 1, c - 1] = True
                if c + 1 < BW.shape[1] and BW[r + 1, c + 1] == True and (r + 1, c + 1) not in idxlist:
                    idxlist.append((r + 1, c + 1))
                    mask[r + 1, c + 1] = True
            k += 1
        a = np.array(idxlist, dtype=np.int64)
        PixelIdxList.append((a[:, 0].tolist(), a[:, 1].tolist()))
    return PixelIdxList


def conjugate_gradient(A, b, x, max_iter=15, tol=1e-4):
    r = b - A(x)
    p = r.clone()
    rs_old = torch.sum(r ** 2)

    for i in range(max_iter):
        Ap = A(p)
        alpha = rs_old / torch.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r ** 2)

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def psf_to_otf(psf, shape):
    psf_pad = torch.zeros(shape, dtype=psf.dtype, device=psf.device)
    psf_pad[:psf.shape[0], :psf.shape[1]] = psf
    psf_pad = torch.roll(psf_pad, -math.ceil(psf.shape[0] / 2), dims=0)
    psf_pad = torch.roll(psf_pad, -math.ceil(psf.shape[1] / 2), dims=1)
    otf = torch.fft.fft2(psf_pad)
    return otf


def adjust_kernel_center(kernel):
    device = kernel.device
    kernel = kernel.cpu().numpy()
    X, Y = np.arange(kernel.shape[1]).reshape(1, -1), np.arange(kernel.shape[0]).reshape(-1, 1)
    kernel_sum = kernel.sum()
    xc1 = np.sum(kernel * X) / kernel_sum
    yc1 = np.sum(kernel * Y) / kernel_sum
    xc2 = (kernel.shape[1] - 1) / 2
    yc2 = (kernel.shape[0] - 1) / 2
    xshift = xc2 - xc1
    yshift = yc2 - yc1
    return torch.tensor(shift(kernel, [yshift, xshift]), device=device)
