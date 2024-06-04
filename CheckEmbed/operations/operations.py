# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

from abc import ABC, abstractmethod
from typing import Any

class Operation(ABC):
    """
    Abstract base class that defines the interface for all operations to be performed on the embeddings/samples.
    """

    def __init__(self, result_dir_path: str) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        """
        self.result_dir_path = result_dir_path

    @abstractmethod
    def execute(self, custom_inputs: Any = None) -> Any:
        """
        Execute the operation on the embeddings/samples.

        :param custom_inputs: The custom inputs for the operation. Defaults to None.
        :type custom_inputs: Any
        """
        pass
