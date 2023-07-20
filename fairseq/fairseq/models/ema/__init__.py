# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from .ema import EMA
from .ema_subset import EMASubset


def build_ema(model, cfg, device, parameters=None):
    if parameters is not None:
        return EMASubset(model, cfg, device, parameters)
    return EMA(model, cfg, device)


# automatically import any Python files in the models/ema/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.models.ema." + file_name)
