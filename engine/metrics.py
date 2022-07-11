# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""metrics for nerf"""

import mindspore as md

__all__ = ["mse", "psnr_from_mse"]


def mse(im1, im2):
    """MSE between two images."""
    return md.numpy.mean((im1 - im2)**2)


def psnr_from_mse(v):
    """Convert MSE to PSNR."""
    return -10.0 * (md.numpy.log(v) / md.numpy.log(md.Tensor([10.0])))
