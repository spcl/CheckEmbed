# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

from abc import abstractmethod
from typing import Any

from CheckEmbed.operations import Operation

class PlotOperation(Operation):
    """
    Abstract base class that defines the interface for all operations that plot data.
    """

    def __init__(self, result_dir_path: str, data_dir_path: str) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        :param data_dir_path: The path to the directory where the data is stored.
        :type data_dir_path: str
        """
        super().__init__(result_dir_path)
        self.data_dir_path = data_dir_path

    @abstractmethod
    def execute(self, custom_inputs: Any = None) -> None:
        """
        Plot the data.

        :param custom_inputs: The custom inputs for the operation. Defaults to None.
        :type custom_inputs: Any
        """
        pass