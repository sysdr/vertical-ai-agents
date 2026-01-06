"""
Test Pydantic schema validation
"""
import pytest
from pydantic import ValidationError
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from schemas import WeatherToolInput, TimeToolInput, SearchToolInput

def test_weather_valid():
    """Test valid weather input"""
    data = WeatherToolInput(location="Paris", unit="celsius")
    assert data.location == "Paris"
    assert data.unit == "celsius"

def test_weather_invalid_unit():
    """Test invalid temperature unit"""
    with pytest.raises(ValidationError):
        WeatherToolInput(location="Paris", unit="kelvin")

def test_weather_invalid_location():
    """Test location with invalid characters"""
    with pytest.raises(ValidationError):
        WeatherToolInput(location="Paris<script>alert()</script>")

def test_time_valid():
    """Test valid time input"""
    data = TimeToolInput(timezone="UTC", format="24h")
    assert data.timezone == "UTC"
    assert data.format == "24h"

def test_search_valid():
    """Test valid search input"""
    data = SearchToolInput(query="Pydantic validation", max_results=5)
    assert data.query == "Pydantic validation"
    assert data.max_results == 5

def test_search_invalid_max_results():
    """Test max_results out of range"""
    with pytest.raises(ValidationError):
        SearchToolInput(query="test", max_results=25)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
