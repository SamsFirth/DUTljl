"""
Utilities for defining Conv2D problem sizes for testing.

This file was ported from the C++ version in test/unit/conv/device/conv2d_problems.h
"""
from cutlass_library import ConvMode
import cutlass
from cutlass.shape import Conv2DProblemSize

class TestbedConv2dProblemSizes:

    def __init__(self, minimum_channel_size: int):
        conv2d_default_sizes = self.initialize_conv2d_default_sizes(minimum_channel_size)
        conv2d_rigorous_sizes = self.initialize_conv2d_rigorous_sizes(minimum_channel_size)
        conv2d_resnet50_sizes = self.initialize_conv2d_resnet50_sizes(1)
        conv2d_resnet50_sizes_perf = self.initialize_conv2d_resnet50_sizes(34)
        grouped_sizes = self.initialize_conv2d_grouped_sizes()
        self.all = []
        for size_list in [conv2d_default_sizes, conv2d_rigorous_sizes, conv2d_resnet50_sizes, conv2d_resnet50_sizes_perf, grouped_sizes]:
            for size in size_list:
                if size.C // size.groups % minimum_channel_size == 0:
                    self.all.append(size)

    def initialize_conv2d_default_sizes(self, minimum_channel_size):
        conv2d_default_sizes = []
        conv2d_default_sizes.append(Conv2DProblemSize(1, 1, 1, minimum_channel_size, 8, 1, 1, minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 1, 8, minimum_channel_size, 8, 1, 3, minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 7, 8, minimum_channel_size, 8, 3, 3, minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 7, 9, minimum_channel_size, 8, 4, 4, minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(2, 7, 9, minimum_channel_size, 8, 5, 5, minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(3, 7, 9, minimum_channel_size, 8, 6, 5, minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(3, 7, 9, minimum_channel_size, 8, 6, 6, minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(3, 7, 9, minimum_channel_size, 8, 7, 7, minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 11, 7, minimum_channel_size, 8, 1, 1, minimum_channel_size, 0, 0, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 11, 7, minimum_channel_size, 8, 3, 3, minimum_channel_size, 1, 1, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 13, 11, minimum_channel_size, 8, 1, 1, minimum_channel_size, 1, 1, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 17, 19, minimum_channel_size, 16, 2, 2, minimum_channel_size, 1, 1, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 23, 5, minimum_channel_size, 16, 3, 3, minimum_channel_size, 1, 1, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 13, 17, 8, 24, 3, 3, 8, 0, 0, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 23, 21, 8, 24, 3, 3, 8, 1, 1, 3, 3, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 20, 24, 8, 40, 3, 3, 8, 3, 3, 3, 3, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 15, 19, 160, 224, 1, 1, 160, 0, 0, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 19, 37, 160, 224, 3, 3, 160, 1, 1, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 16, 16, 160, 224, 2, 3, 160, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 23, 21, 128, 224, 3, 3, 128, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 29, 37, 160, 224, 5, 5, 160, 2, 2, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 15, 19, 32 + minimum_channel_size, 96, 3, 3, 32 + minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 16, 24, 64 + minimum_channel_size, 96, 3, 3, 64 + minimum_channel_size, 1, 1, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 13, 16, 288, 160, 5, 5, 288, 2, 2, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 55, 51, 256, 512, 1, 1, 256, 0, 0, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 71, 80, 32, 64, 5, 5, 32, 2, 2, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 224, 224, 8, 64, 7, 7, 8, 3, 3, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 27, 23, 256, 512, 3, 3, 256, 0, 0, 3, 3, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 27, 31, 256, 512, 3, 3, 256, 5, 7, 3, 4, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 27, 35, 256, 512, 7, 5, 256, 11, 7, 3, 5, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 27, 27, 256, 512, 3, 3, 256, 1, 1, 1, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 27, 27, 256, 512, 3, 3, 256, 1, 1, 2, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(3, 28, 28, 256, 256, 2, 2, 256, 0, 0, 2, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(1, 32, 32, 16, 32, 3, 3, 16, 1, 1, 6, 2, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(32, 24, 32, 32, 32, 1, 2, 32, 0, 0, 1, 1, 1, 1))
        conv2d_default_sizes.append(Conv2DProblemSize(4, 2, 3, 256, 328, 3, 5, 256, 1, 1, 1, 1, 1, 1))
        return conv2d_default_sizes

    def initialize_conv2d_rigorous_sizes(self, minimum_channel_size):
        sizes = []
        if False:
            sizes.append(Conv2DProblemSize.from_sizes((1, 124, 224, 2 * minimum_channel_size), (24, 7, 7, 2 * minimum_channel_size)))
            sizes.append(Conv2DProblemSize.from_sizes((1, 233, 35, minimum_channel_size), (24, 7, 5, minimum_channel_size)))
        return sizes

    def initialize_conv2d_resnet50_sizes(self, batch_size):
        conv2d_problem_vector = []
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 56, 56, 64, 256, 1, 1, 64, 0, 0, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 56, 56, 64, 64, 1, 1, 64, 0, 0, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 56, 56, 64, 64, 3, 3, 64, 1, 1, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 56, 56, 256, 64, 1, 1, 256, 0, 0, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 56, 56, 256, 512, 1, 1, 256, 0, 0, 2, 2, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 56, 56, 256, 128, 1, 1, 256, 0, 0, 2, 2, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 28, 28, 128, 128, 3, 3, 128, 1, 1, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 28, 28, 128, 512, 1, 1, 128, 0, 0, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 28, 28, 512, 128, 1, 1, 512, 0, 0, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 28, 28, 512, 1024, 1, 1, 512, 0, 0, 2, 2, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 28, 28, 512, 256, 1, 1, 512, 0, 0, 2, 2, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 14, 14, 256, 256, 3, 3, 256, 1, 1, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 14, 14, 256, 1024, 1, 1, 256, 0, 0, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 14, 14, 1024, 256, 1, 1, 1024, 0, 0, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 14, 14, 1024, 2048, 1, 1, 1024, 0, 0, 2, 2, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 14, 14, 1024, 512, 1, 1, 1024, 0, 0, 2, 2, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 7, 7, 512, 512, 3, 3, 512, 1, 1, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 7, 7, 512, 2048, 1, 1, 512, 0, 0, 1, 1, 1, 1))
        conv2d_problem_vector.append(Conv2DProblemSize(batch_size, 7, 7, 2048, 512, 1, 1, 2048, 0, 0, 1, 1, 1, 1))
        return conv2d_problem_vector

    def initialize_conv2d_grouped_sizes(self):
        threadblock_n = 128
        threadblock_k = 32
        sizes = []
        for cta_per_group_k in range(1, 4):
            for groups in range(2, 5):
                conv_k = cta_per_group_k * threadblock_n * groups
                sizes.append(Conv2DProblemSize(1, 8, 8, threadblock_k * 2 * groups, conv_k, 3, 3, threadblock_k * 2, 1, 1, 1, 1, 1, 1, ConvMode.CrossCorrelation, 1, groups))
        sizes.append(Conv2DProblemSize(1, 8, 8, threadblock_k, threadblock_n * 2, 3, 3, threadblock_k // 2, 1, 1, 1, 1, 1, 1, ConvMode.CrossCorrelation, 1, 2))
        sizes.append(Conv2DProblemSize(1, 56, 56, 696, 768, 3, 3, 232, 1, 1, 2, 2, 1, 1, ConvMode.CrossCorrelation, 1, 3))
        sizes.append(Conv2DProblemSize(1, 14, 14, 1392, 1536, 3, 3, 232, 1, 1, 1, 1, 1, 1, ConvMode.CrossCorrelation, 1, 3))
        sizes.append(Conv2DProblemSize(1, 8, 8, threadblock_k * 4, threadblock_n, 3, 3, threadblock_k * 2, 1, 1, 1, 1, 1, 1, ConvMode.CrossCorrelation, 1, 2))
        sizes.append(Conv2DProblemSize(1, 8, 8, threadblock_k, threadblock_n, 3, 3, threadblock_k // 2, 1, 1, 1, 1, 1, 1, ConvMode.CrossCorrelation, 1, 2))
        sizes.append(Conv2DProblemSize(1, 8, 8, threadblock_k * 8, threadblock_n // 2, 3, 3, threadblock_k * 2, 1, 1, 1, 1, 1, 1, ConvMode.CrossCorrelation, 1, 4))
        sizes.append(Conv2DProblemSize(1, 8, 8, threadblock_k * 2, threadblock_n // 2, 3, 3, threadblock_k // 2, 1, 1, 1, 1, 1, 1, ConvMode.CrossCorrelation, 1, 4))
        return sizes