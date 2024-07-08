import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import numpy as np
import math

from .utils import *


class OID:
    def __init__(self, blurred_image, kernels_size, device, latent_type='gray') -> None:
        self.kernels_size = kernels_size
        self.scales = [k / kernels_size[-1] for k in kernels_size]
        self.device = device

        self.latent_type = latent_type
        self.set_image(blurred_image)

        self.weight_grad_I = 4e-4
        self.weight_K = 5.0
        self.weight_W = 1.8e-3
        self.weight_entropy_W = 2e-4

        # self.weight_grad_I = 1.93e-3
        # self.weight_K = 4.27
        # self.weight_W = 1.97e-3
        # self.weight_entropy_W = 6.32e-4

        self.epoch = 10
        self.epoch_I = 4
        self.epoch_K = 4

    def set_image(self, blurred_image):
        self.num_channels, *self.image_size = blurred_image.shape
        # self.pad = pad
        # self.blurred_image = torch.nn.functional.pad(
        #     torch.tensor(blurred_image, dtype=torch.float32, device=self.device), (pad, pad, pad, pad), mode='reflect')
        self.blurred_image = torch.tensor(blurred_image, dtype=torch.float32, device=self.device)

        # worse_image = torch.nn.functional.interpolate(self.blurred_image.unsqueeze(0), scale_factor=self.scales[0], mode='bilinear').squeeze(0)

        if self.latent_type == 'gray':
            self.B = torch.nn.functional.interpolate(self.blurred_image.mean(dim=0, keepdim=True).unsqueeze(0),
                                                     size=self.compute_size(self.scales[0]), mode='bilinear').squeeze(0)
        elif self.latent_type == 'color':
            self.B = torch.nn.functional.interpolate(self.blurred_image.unsqueeze(0),
                                                     size=self.compute_size(self.scales[0]), mode='bilinear').squeeze(0)

        self.I = self.B.clone()
        self.W = torch.ones(1, self.I.shape[1], self.I.shape[2], device=self.device)
        self.K = torch.zeros(1, self.kernels_size[0], self.kernels_size[0], dtype=torch.float32, device=self.device)
        self.K[0, self.kernels_size[0] // 2, self.kernels_size[0] // 2 - 1:self.kernels_size[0] // 2 + 1] = 0.5
        # self.K = self.gaussian_kernel(self.kernels_size[0], self.kernels_size[0] * 0.5).to(self.device).unsqueeze(0)

    # def gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
    #     """Generate a 2D Gaussian kernel."""
    #     # Create a grid of (x, y) coordinates
    #     x_coord = torch.arange(size)
    #     x_grid = x_coord.repeat(size).view(size, size)
    #     y_grid = x_grid.t()

    #     xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    #     mean = (size - 1) / 2.
    #     variance = sigma ** 2.

    #     # Calculate the 2D Gaussian kernel
    #     gaussian_kernel = (1. / (2. * torch.pi * variance)) * \
    #         torch.exp(
    #         -torch.sum((xy_grid - mean) ** 2., dim=-1) /
    #         (2 * variance)
    #     )

    #     # Normalize the kernel
    #     gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    #     return gaussian_kernel

    def compute_blurred_image(self, latent, kernel):
        computed_image = torch.conv2d(latent, kernel.expand(latent.shape[0], -1, -1, -1), padding=kernel.shape[-1] // 2, groups=latent.shape[0])
        # return torch.clamp(computed_image, 0.0, 1.0)
        return computed_image

    def compute_size(self, scale):
        return (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

    def optimize_W(self):
        self.W = torch.sigmoid(-((self.B.mean(0, keepdim=True) - self.compute_blurred_image(self.I, self.K).mean(0, keepdim=True))
                               ** 2 - self.weight_W) / self.weight_entropy_W).detach()

    def regularize_K(self):
        result = bwconncomp_2d(np.array(self.K.squeeze(0).detach().cpu() != 0.0, dtype=np.bool8))
        for r in result:
            if torch.sum(self.K[:, *r]) < 0.1:  # 原文为 0.1
                self.K[:, *r] = 0.0
                pass
        self.K[self.K < 0.0] = 0.0
        self.K /= self.K.sum()
        # self.K.data /= self.K.sum()

    def grad_image(self, image):
        return image[:, :, :-1] - image[:, :, 1:], image[:, :-1, :] - image[:, 1:, :]

    def optimize(self, callback=None):
        for epoch in range(self.epoch):
            for step in range(self.epoch_I):
                # prev_I = self.I.clone()
                # prev_loss = 1e10

                grad_x_I, grad_y_I = self.grad_image(self.I.detach())
                P_x = torch.max(torch.abs(grad_x_I), torch.tensor(1e-2)) ** -1.2
                P_y = torch.max(torch.abs(grad_y_I), torch.tensor(1e-2)) ** -1.2  # eps=1e-5就不行 真离谱

                def latent_loss(latent):
                    deblur_loss = torch.sum(self.W * (self.B - self.compute_blurred_image(latent, self.K)) ** 2)

                    grad_x_latent, grad_y_latent = self.grad_image(latent)
                    DI_loss = self.weight_grad_I * (torch.sum(P_x * grad_x_latent ** 2) +
                                                    torch.sum(P_y * grad_y_latent ** 2))
                    loss = deblur_loss + DI_loss
                    return loss

                grad_I = torch.func.jacrev(latent_loss)
                b = -grad_I(torch.zeros_like(self.I))
                self.I.data = conjugate_gradient(lambda x: grad_I(x) + b, b, self.I.clone(), 25)

                self.I.data.clamp_(0, 1)
                self.optimize_W()

                if callback is not None:
                    callback(**{
                        'epoch': epoch,
                        'step': step,
                        'type': 'I',
                        'loss': latent_loss(self.I).item()
                    })

            for step in range(self.epoch_K):
                self.optimize_W()

                grad_x_B, grad_y_B = self.grad_image(self.B)
                grad_x_I, grad_y_I = self.grad_image(self.I)
                # grad_I = torch.sqrt(grad_x_I[:, :-1, :] ** 2 + grad_y_I[:, :, :-1] ** 2)
                # grad_x_I[:, :-1, :][grad_I < 0.0188] = 0.0
                # grad_y_I[:, :, :-1][grad_I < 0.0188] = 0.0
                W_x = torch.sigmoid(-((grad_x_B.mean(0, keepdim=True) - self.compute_blurred_image(grad_x_I, self.K).mean(0, keepdim=True))
                                      ** 2 - self.weight_W) / self.weight_entropy_W)
                W_y = torch.sigmoid(-((grad_y_B.mean(0, keepdim=True) - self.compute_blurred_image(grad_y_I, self.K).mean(0, keepdim=True))
                                      ** 2 - self.weight_W) / self.weight_entropy_W)

                def kernel_loss(kerenl):
                    # self.K.data /= (self.K.sum() + 1e-5)
                    # deblur_loss = torch.sum(self.W * (self.B - self.compute_blurred_image(self.I, kerenl)) ** 2)
                    deblur_loss = torch.sum(W_x * (grad_x_B - self.compute_blurred_image(grad_x_I, kerenl)) ** 2) + \
                        torch.sum(W_y * (grad_y_B - self.compute_blurred_image(grad_y_I, kerenl)) ** 2)
                    K_loss = self.weight_K * torch.sum(kerenl ** 2)
                    return deblur_loss + K_loss

                grad_K = torch.func.jacrev(kernel_loss)
                b = -grad_K(torch.zeros_like(self.K))
                self.K = conjugate_gradient(lambda x: grad_K(x) + b, b, self.K, 21)

                self.K[self.K < 0.05 * self.K.max()] = 0.0
                self.K /= self.K.sum()

                if callback is not None:
                    callback(**{
                        'epoch': epoch,
                        'step': step,
                        'type': 'K',
                        'loss': kernel_loss(self.K).item()
                    })
            self.regularize_K()

    def coarse_deblur(self, beta, lambd, kappa, beta_max, callback=None):
        for epoch in range(4):
            pad_h, pad_w = self.K.shape[1] - 1, self.K.shape[2] - 1
            image_padded = torch.nn.functional.pad(self.B, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            # image_padded = self.B

            S = image_padded.clone()
            fx = torch.tensor([[1, -1]], device=self.device)
            fy = torch.tensor([[1], [-1]], device=self.device)
            C, H, W = image_padded.shape
            otfFx = psf_to_otf(fx, (H, W))
            otfFy = psf_to_otf(fy, (H, W))
            kernel_hat = psf_to_otf(self.K[0].flip(0).flip(1), (H, W)).unsqueeze(0)
            density_kernel = torch.abs(kernel_hat) ** 2
            Denormin2 = torch.abs(otfFx) ** 2 + torch.abs(otfFy) ** 2
            if C > 1:
                Denormin2 = Denormin2.unsqueeze(0).expand(C, H, W)
                kernel_hat = kernel_hat.expand(C, H, W)
                density_kernel = density_kernel.expand(C, H, W)
            Normin1 = torch.conj(kernel_hat) * torch.fft.fft2(S)

            # beta = kappa * lambd
            beta_loop = beta
            while beta_loop < beta_max:
                Denormin = density_kernel + beta_loop * Denormin2
                h = torch.cat((S[:, :, 1:] - S[:, :, :-1], S[:, :, :1] - S[:, :, -1:]), dim=2)
                v = torch.cat((S[:, 1:, :] - S[:, :-1, :], S[:, :1, :] - S[:, -1:, :]), dim=1)
                t = (h ** 2 + v ** 2).sum(dim=0, keepdim=True) < lambd / beta_loop
                h[t.expand_as(h)] = 0.0
                v[t.expand_as(v)] = 0.0
                Normin2 = torch.cat((h[:, :, -1:] - h[:, :, :1], -h[:, :, 1:] + h[:, :, :-1]), dim=2)
                Normin2 += torch.cat((v[:, -1:, :] - v[:, :1, :], -v[:, 1:, :] + v[:, :-1, :]), dim=1)
                FS = (Normin1 + beta_loop * torch.fft.fft2(Normin2)) / Denormin
                S = torch.real(torch.fft.ifft2(FS))
                S.data.clamp_(0, 1)
                # import matplotlib.pyplot as plt
                # plt.imshow(S.cpu().numpy().transpose(1, 2, 0), cmap='gray')
                beta_loop *= kappa

            self.I = S[:, pad_h + 1:pad_h + self.I.shape[1] + 1, pad_w + 1:pad_w + self.I.shape[2] + 1]
            # self.I = S
            # self.I.data.clamp_(0, 1)
            if callback is not None:
                callback(**{
                    'epoch': epoch,
                    'step': 0,
                    'type': 'I'
                })

            for step in range(1):
                grad_x_B, grad_y_B = self.grad_image(self.B)
                grad_x_I, grad_y_I = self.grad_image(self.I)
                # # grad_I = torch.sqrt(grad_x_I[:, :-1, :] ** 2 + grad_y_I[:, :, :-1] ** 2)
                # # grad_x_I[:, :-1, :][grad_I < 0.0188] = 0.0
                # # grad_y_I[:, :, :-1][grad_I < 0.0188] = 0.0
                # W_x = torch.sigmoid(-((grad_x_B.mean(0, keepdim=True) - self.compute_blurred_image(grad_x_I, self.K).mean(0, keepdim=True))
                #                       ** 2 - self.weight_W) / self.weight_entropy_W)
                # W_y = torch.sigmoid(-((grad_y_B.mean(0, keepdim=True) - self.compute_blurred_image(grad_y_I, self.K).mean(0, keepdim=True))
                #                       ** 2 - self.weight_W) / self.weight_entropy_W)

                def kernel_loss(kerenl):
                    # self.K.data /= (self.K.sum() + 1e-5)
                    # deblur_loss = torch.sum(self.W * (self.B - self.compute_blurred_image(self.I, kerenl)) ** 2)
                    deblur_loss = torch.sum((grad_x_B - self.compute_blurred_image(grad_x_I, kerenl)) ** 2) + \
                        torch.sum((grad_y_B - self.compute_blurred_image(grad_y_I, kerenl)) ** 2)
                    # deblur_loss = torch.sum((self.B - self.compute_blurred_image(self.I, kerenl)) ** 2)
                    K_loss = self.weight_K * torch.sum(kerenl ** 2)
                    return deblur_loss + K_loss

                grad_K = torch.func.jacrev(kernel_loss)
                b = -grad_K(torch.zeros_like(self.K))
                self.K = conjugate_gradient(lambda x: grad_K(x) + b, b, self.K, 20)

                self.K[self.K < 0.05 * self.K.max()] = 0.0
                self.K /= self.K.sum()

                if callback is not None:
                    callback(**{
                        'epoch': epoch,
                        'step': step,
                        'type': 'K',
                        'loss': kernel_loss(self.K).item()
                    })

            self.regularize_K()
            # self.K = adjust_kernel_center(self.K)

    # def conjugate_gradient(self, A, b, x, max_iter=15, tol=1e-4):
    #     r = b - A(x)
    #     p = r.clone()
    #     rs_old = torch.sum(r ** 2)

    #     for i in range(max_iter):
    #         Ap = A(p)
    #         alpha = rs_old / torch.sum(p * Ap)
    #         x = x + alpha * p
    #         r = r - alpha * Ap
    #         rs_new = torch.sum(r ** 2)

    #         if torch.sqrt(rs_new) < tol:
    #             break

    #         p = r + (rs_new / rs_old) * p
    #         rs_old = rs_new

    #     return x

    def train(self, callback=None):
        for i, scale in enumerate(self.scales):
            if i > 0:
                if self.latent_type == 'gray':
                    self.B = torch.nn.functional.interpolate(self.blurred_image.mean(dim=0, keepdim=True).unsqueeze(0),
                                                             size=self.compute_size(self.scales[i]), mode='bilinear').squeeze(0)
                elif self.latent_type == 'color':
                    self.B = torch.nn.functional.interpolate(self.blurred_image.unsqueeze(0),
                                                             size=self.compute_size(self.scales[i]), mode='bilinear').squeeze(0)

                self.I = torch.nn.functional.interpolate(self.I.unsqueeze(0),
                                                         size=self.compute_size(self.scales[i]),
                                                         mode='bilinear').squeeze(0)
                self.W = torch.nn.functional.interpolate(self.W.unsqueeze(0), size=self.I.shape[-2:], mode='bilinear').squeeze(0)
                self.K = torch.nn.functional.interpolate(self.K.unsqueeze(0),
                                                         size=(self.kernels_size[i], self.kernels_size[i]),
                                                         mode='bilinear').squeeze(0)
                self.regularize_K()
            if i < len(self.scales) - 1:
                self.B = transforms.functional.gaussian_blur(self.B, 5, 1)
                self.coarse_deblur(beta=8e-3, lambd=4e-3, kappa=2, beta_max=1e5, callback=callback)
            else:
                self.optimize(callback)
            self.K = adjust_kernel_center(self.K.squeeze(0)).unsqueeze(0)
            # W_loss = self.weight_W * torch.sum(torch.abs(1 - self.W))
            # W_entropy_loss = self.weight_entropy_W = torch.nn.functional.binary_cross_entropy_with_logits(self.W, self.W, reduction='sum')

    def estimate_latent(self, weight_grad_I=3e-4, pad=None, callback=None):
        if pad is None:
            pad = self.kernels_size[-1] * 2
        # print(torch.mean(self.K ** 2))
        c, h, w = self.blurred_image.shape
        blurred_image = torch.nn.functional.pad(self.blurred_image, (pad, pad, pad, pad), mode='reflect')
        latent_image = blurred_image.clone()
        # grad_x_I, grad_y_I = self.grad_image(latent_image)
        # P_x = torch.max(torch.abs(grad_x_I), torch.tensor(1e-2)) ** -1.2
        # P_y = torch.max(torch.abs(grad_y_I), torch.tensor(1e-2)) ** -1.2  # eps=1e-5就不行 真离谱
        W = torch.sigmoid(-((blurred_image - self.compute_blurred_image(latent_image, self.K)) ** 2 - self.weight_W) / self.weight_entropy_W)

        # max_blur = torch.max(blurred_image)
        # min_blur = torch.min(blurred_image)
        # W[blurred_image == min_blur] = 0.0
        # W[blurred_image == max_blur] = 0.0
        # W[W <= 0] = 1e-16
        # W[W >= 1] = 1-1e-16

        def latent_loss(latent):
            deblur_loss = torch.sum(W * (blurred_image - self.compute_blurred_image(latent, self.K)) ** 2)

            grad_x_latent, grad_y_latent = self.grad_image(latent)
            DI_loss = weight_grad_I * (torch.sum(grad_x_latent ** 2) +
                                       torch.sum(grad_y_latent ** 2))
            loss = deblur_loss + DI_loss
            return loss

        grad_I = torch.func.jacrev(latent_loss)
        b = -grad_I(torch.zeros_like(latent_image))
        latent_image = conjugate_gradient(lambda x: grad_I(x) + b, b, latent_image, 25)

        latent_image.data.clamp_(0, 1)
        W = torch.sigmoid(-((blurred_image - self.compute_blurred_image(latent_image, self.K)) ** 2 - self.weight_W) / self.weight_entropy_W)

        # max_blur = torch.max(blurred_image)
        # min_blur = torch.min(blurred_image)
        # W[blurred_image == min_blur] = 0.0
        # W[blurred_image == max_blur] = 0.0
        # W[W <= 0] = 1e-8
        # W[W >= 1] = 1-1e-8

        if not callback is None:
            callback(np.array(latent_image[:, pad:h + pad, pad:w + pad].cpu()).transpose(1, 2, 0), 0)
        # W = self.W
        for epoch in range(15):
            # prev_I = self.I.clone()
            # prev_loss = 1e10
            grad_x_I, grad_y_I = self.grad_image(latent_image)
            P_x = torch.max(torch.abs(grad_x_I), torch.tensor(1e-2)) ** -1.2
            P_y = torch.max(torch.abs(grad_y_I), torch.tensor(1e-2)) ** -1.2  # eps=1e-5就不行 真离谱

            def latent_loss(latent):
                deblur_loss = torch.sum(W * (blurred_image - self.compute_blurred_image(latent, self.K)) ** 2)

                grad_x_latent, grad_y_latent = self.grad_image(latent)
                DI_loss = weight_grad_I * (torch.sum(P_x * grad_x_latent ** 2) +
                                           torch.sum(P_y * grad_y_latent ** 2))
                loss = deblur_loss + DI_loss
                return loss

            grad_I = torch.func.jacrev(latent_loss)
            b = -grad_I(torch.zeros_like(latent_image))
            latent_image = conjugate_gradient(lambda x: grad_I(x) + b, b, latent_image, 25)

            latent_image.data.clamp_(0, 1)
            W = torch.sigmoid(-((blurred_image - self.compute_blurred_image(latent_image, self.K)) ** 2 - self.weight_W) / self.weight_entropy_W)

            if not callback is None:
                callback(np.array(latent_image[:, pad:h + pad, pad:w + pad].cpu()).transpose(1, 2, 0), epoch)

        return np.array(latent_image.cpu()).transpose(1, 2, 0)
