import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import subprocess

import src.utils # Assuming orchestration.py is in the same directory


class TestUtils(unittest.TestCase):

    @patch('src.utils.generate_requirements_txt')
    @patch('src.install_requirements_from_file')
    def test_setup(self, mock_install, mock_generate_requirements_txt):
        # Mock the generate_requirements_txt function to simulate creating the file
        mock_generate_requirements_txt.return_value = None  # Simulate success
        mock_install.return_value = None  # Simulate success

        # Run setup with a parameter
        src.utils.setup("Test Parameter")

        # Check that generate_requirements_txt was called with the expected directory
        mock_generate_requirements_txt.assert_called_with('.')

        # Check that install_requirements_from_file was called
        mock_install.assert_called_with(requirements_file='requirements.txt')

    @patch('subprocess.check_call')
    def test_install_requirements_from_file(self, mock_check_call):
        # Simulate successful package installation
        mock_check_call.return_value = None

        # Run the install function
        src.utils.install_requirements_from_file('requirements.txt')

        # Ensure subprocess.check_call was called with the correct arguments
        mock_check_call.assert_called_with([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

    @patch('orchestration.train_and_test.main')
    def test_of_we_go(self, mock_train_and_test):
        # Simulate running the neural network training
        mock_train_and_test.return_value = None

        # Run the training process
        src.utils.of_we_go("Test Parameter")

        # Ensure that the main training function was called
        mock_train_and_test.assert_called()


if __name__ == '__main__':
    unittest.main()
