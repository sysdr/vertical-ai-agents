"""
Basic environment tests
"""
import pytest
import sys

def test_python_version():
    """Verify Python 3.12+"""
    assert sys.version_info >= (3, 12), "Python 3.12+ required"

def test_imports():
    """Verify core packages import successfully"""
    import fastapi
    import pydantic
    import numpy
    import pandas
    import google.generativeai
    
    # If we get here, all imports succeeded
    assert True

def test_environment_setup():
    """Verify basic environment is functional"""
    import os
    assert os.path.exists('venv'), "Virtual environment should exist"
    assert os.path.exists('requirements.txt'), "Requirements file should exist"
    assert os.path.exists('.env'), "Environment file should exist"
