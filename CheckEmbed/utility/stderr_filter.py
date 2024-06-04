# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import contextlib
import sys
import tempfile
import os

from typing import Generator, List, Union, Any

class FilteredStderr:
    """
    A class that captures and filters stderr output.

    The class creates a temporary file to capture the stderr stream and filters the stream based on target string(s).
    """

    def __init__(self, target_string: Union[List[str], str]) -> None:
        """
        Initializes a FilteredStderr instance.

        :param target_string: Target string(s) for filtering stderr.
        :type target_string: Union[List[str], str]
        """
        self.target_string = target_string
        self.captured = ""
        self.original_stderr_fd = None
        self.temp_fd = None
        self.temp_file = None

    def start(self) -> None:
        """
        Start capturing stderr and redirecting the stream to a temporary file.
        """
        # Save the original stderr file descriptor
        self.original_stderr_fd = os.dup(2)
        # Create a temporary file and file descriptor to capture stderr
        self.temp_file = tempfile.TemporaryFile(mode='w+')
        self.temp_fd = self.temp_file.fileno()
        # Redirect stderr to the temporary file
        os.dup2(self.temp_fd, 2)

    def stop(self) -> None:
        """
        Stop capturing stderr. Filter the stream for the target string(s) and restore the original stderr file descriptor.
        """
        # Restore the original stderr file descriptor
        os.dup2(self.original_stderr_fd, 2)
        os.close(self.original_stderr_fd)
        self.original_stderr_fd = None
        # Read the captured output
        self.temp_file.seek(0)
        output = self.temp_file.read()
        self.temp_file.close()
        self.temp_fd = None
        self.temp_file = None
        # Filter the output
        for line in output.splitlines():
            # target string can be an array
            if isinstance(self.target_string, str):
                self.target_string = [self.target_string]
            
            captured = False
            for target in self.target_string:
                if target in line:
                    self.captured += line + "\n"
                    captured = True
                    break
            
            if not captured:
                sys.__stderr__.write(line + "\n")


@contextlib.contextmanager
def capture_specific_stderr(custom_target: Union[List[str], str] = None) -> Generator[FilteredStderr, None, None]:
    """
    Context manager that captures and filters the stderr stream.

    :param custom_target: Target string(s) for filtering stderr. If None, default target strings are loaded from a file.
    :type custom_target: Union[List[str], str]
    :return: A FilteredStderr instance, which can be used to access the captured stderr stream.
    :rtype: Generator[FilteredStderr, None, None]
    """
    # load default target strings from file
    if custom_target is None:
        with open(os.path.join(os.path.dirname(__file__), "./stderr_filter.txt"), "r") as file:
            custom_target = file.read().splitlines()
    
    filtered_stderr = FilteredStderr(custom_target)
    filtered_stderr.start()
    try:
        yield filtered_stderr
    finally:
        filtered_stderr.stop()
