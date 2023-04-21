import torch
from pytorch_model_summary import summary
from torch import Tensor
from typing import Tuple, Union, Optional
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d
from torch.nn.modules.activation import ReLU


class DMCNN(Module):

    def __init__(self,
                conv_1_output_dim: int,
                conv_2_output_dim: int,
                conv_3_output_dim: int,
                conv_1_kernel_size: Union[int, Tuple[int, int]],
                conv_2_kernel_size: Union[int, Tuple[int, int]],
                conv_3_kernel_size: Union[int, Tuple[int, int]],
                conv_1_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_2_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_3_stride: Optional[Union[int, Tuple[int, int]]] = 1,
                conv_1_padding: Optional[Union[int, Tuple[int, int]]] = 0,
                conv_2_padding: Optional[Union[int, Tuple[int, int]]] = 0,
                conv_3_padding: Optional[Union[int, Tuple[int, int]]] = 0,
    ) -> None:

        super().__init__()

        self.block_1 = Sequential(
            Conv2d(in_channels=3,
                   out_channels=conv_1_output_dim,
                   kernel_size=conv_1_kernel_size,
                   stride=conv_1_stride,
                   padding=conv_1_padding),
            BatchNorm2d(num_features=conv_1_output_dim),
            ReLU(),
        )

        self.block_2 = Sequential(
            Conv2d(in_channels=conv_1_output_dim,
                   out_channels=conv_2_output_dim,
                   kernel_size=conv_2_kernel_size,
                   stride=conv_2_stride,
                   padding=conv_2_padding),
            BatchNorm2d(num_features=conv_2_output_dim),
            ReLU(),
        )

        self.block_3 = Sequential(
            Conv2d(in_channels=conv_2_output_dim,
                   out_channels=conv_3_output_dim,
                   kernel_size=conv_3_kernel_size,
                   stride=conv_3_stride,
                   padding=conv_3_padding),
            BatchNorm2d(num_features=conv_3_output_dim),
            ReLU(),
        )

    def forward(self, X: Tensor) -> Tensor:
        y = self.block_1(X)
        y = self.block_2(y)
        y = self.block_3(y)

        return y


def main():
    conv_1_output_dim = 128
    conv_2_output_dim = 64
    conv_3_output_dim = 3

    conv_1_kernel_size = 9
    conv_2_kernel_size = 1
    conv_3_kernel_size = 5

    model = DMCNN(
        conv_1_output_dim=conv_1_output_dim,
        conv_2_output_dim=conv_2_output_dim,
        conv_3_output_dim=conv_3_output_dim,
        conv_1_kernel_size=conv_1_kernel_size,
        conv_2_kernel_size=conv_2_kernel_size,
        conv_3_kernel_size=conv_3_kernel_size,
    )

    print(summary(model, torch.rand(64,3,33,33).float()))


if __name__ == "__main__":
    main()