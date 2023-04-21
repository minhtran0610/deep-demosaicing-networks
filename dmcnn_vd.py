import torch
from pytorch_model_summary import summary
from torch import Tensor
from typing import Tuple, Union, Optional
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, init, Dropout2d
from torch.nn.modules.activation import SELU, ReLU

def he_init_weights(m):
    if isinstance(m, Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='selu')
        if m.bias is not None:
            init.zeros_(m.bias)


class DMCNN_VD(Module):

    def __init__(self,
                num_hiddel_blocks: Optional[int] = 20,
                num_conv_channels: Optional[int] = 64,
                conv_kernel_size: Optional[Union[int, Tuple[int, int]]] = 3,
                conv_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_padding: Optional[Union[int, Tuple[int, int]]] = 1,
                dropout: Optional[Union[int, Tuple[int, int]]] = 0.25
    ) -> None:

        super().__init__()

        self.input_block = Sequential(
            Conv2d(in_channels=3,
                   out_channels=num_conv_channels,
                   kernel_size=conv_kernel_size,
                   stride=conv_stride,
                   padding=conv_padding),
            BatchNorm2d(num_features=num_conv_channels),
            SELU(),
        )
        self.input_block.apply(he_init_weights)

        self.hidden_block = Sequential(
            Conv2d(in_channels=num_conv_channels,
                   out_channels=num_conv_channels,
                   kernel_size=conv_kernel_size,
                   stride=conv_stride,
                   padding=conv_padding),
            BatchNorm2d(num_features=num_conv_channels),
            SELU(),
            Dropout2d(p=dropout),
        )
        self.hidden_block.apply(he_init_weights)

        self.output_block = Sequential(
            Conv2d(in_channels=num_conv_channels,
                   out_channels=3,
                   kernel_size=conv_kernel_size,
                   stride=conv_stride,
                   padding=conv_padding),
            BatchNorm2d(num_features=3),
            SELU(),
        )
        self.output_block.apply(he_init_weights)

        self.num_hiddel_blocks = num_hiddel_blocks

    def interpolate_2d_array(self, arr):
        """
        Estiate values at places with 0 in a 2D PyTorch tensor using bilinear interpolation
        """
        device = arr.device
        H, W = arr.shape

        # Initialize the interpolated tensor to the original tensor
        interpolated_arr = arr.clone()

        # Create a binary mask tensor indicating the zero pixels
        zero_mask = arr == 0

        # Pad the tensor with zeros along the borders
        padded_arr = torch.nn.functional.pad(arr, (1, 1, 1, 1), mode='constant', value=0)

        # Extract the neighboring pixel values using tensor slicing
        left = padded_arr[1:H+1, :W][zero_mask]
        right = padded_arr[1:H+1, 2:W+2][zero_mask]
        top = padded_arr[:H, 1:W+1][zero_mask]
        bottom = padded_arr[2:H+2, 1:W+1][zero_mask]

        # Compute the missing values
        neighbor_points = torch.stack([left, right, top, bottom])
        num_nonzero_neighbors = (neighbor_points>0).sum(dim=0)
        neighbor_sum = neighbor_points.sum(dim=0)
        neighbor_mean = torch.where(num_nonzero_neighbors > 0, neighbor_sum / num_nonzero_neighbors.float(), torch.tensor([0.]).to(device))

        # Fill in the interpolated tensor with the estimated values
        interpolated_arr[zero_mask] = neighbor_mean

        return interpolated_arr

    def interpolate_zeros(self, bayer_imgs):
        """
        Estimate values at places with 0 in PyTorch tensors representing a batch of Bayer images of shape (B, 3, H, W) using bilinear interpolation
        """
        device = bayer_imgs.device
        B, C, _, _ = bayer_imgs.shape
        
        # Initialize the interpolated tensor to the original tensor
        interpolated_bayer_imgs = bayer_imgs.clone()
        
        # Blinear interpolation of each 2D array in the batch
        for i in range(B):
            for j in range(C):
                interpolated_bayer_imgs[i,j] = self.interpolate_2d_array(bayer_imgs[i,j]).to(device)
        
        return interpolated_bayer_imgs

    def forward(self, X: Tensor) -> Tensor:
        y = self.input_block(X)

        for i in range(self.num_hiddel_blocks-2):
            y = self.hidden_block(y)
        
        y = self.output_block(y) + self.interpolate_zeros(X)
        return y


def main():
    model = DMCNN_VD(num_hiddel_blocks=20)
    print(summary(model, torch.rand(64,3,33,33).float()))


if __name__ == "__main__":
    main()