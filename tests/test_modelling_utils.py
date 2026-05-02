"""
Tests for modelling utility functions.
"""

import pytest

from sdm.commands.modelling.utils import clean_string, get_model_id


class TestCleanString:
    """Test the clean_string function."""
    
    def test_basic_cleaning(self):
        """Test basic string cleaning functionality."""
        assert clean_string("Hello World") == "hello_world"
        assert clean_string("Test String") == "test_string"
    
    def test_whitespace_handling(self):
        """Test whitespace handling."""
        # Note: clean_string replaces spaces with underscores but doesn't collapse multiple spaces
        assert clean_string("  hello  world  ") == "hello__world"
        assert clean_string("hello   world") == "hello___world"
    
    def test_leading_trailing_underscores(self):
        """Test removal of leading/trailing underscores."""
        assert clean_string("_hello_world_") == "hello_world"
        assert clean_string("  _test_  ") == "test"
    
    def test_lowercase_disabled(self):
        """Test with lowercase disabled."""
        assert clean_string("Hello World", lowercase=False) == "Hello_World"
        assert clean_string("TEST", lowercase=False) == "TEST"
    
    def test_empty_string(self):
        """Test with empty string."""
        assert clean_string("") == ""
        assert clean_string("   ") == ""


class TestGetModelId:
    """Test the get_model_id function."""
    
    def test_single_part(self):
        """Test with single part."""
        assert get_model_id(["Myotis daubentonii"]) == "myotis_daubentonii"
        assert get_model_id(["In flight"]) == "in_flight"
    
    def test_multiple_parts(self):
        """Test with multiple parts."""
        assert get_model_id(["Myotis daubentonii", "In flight"]) == "myotis_daubentonii_in_flight"
        assert get_model_id(["Nyctalus noctula", "Roost"]) == "nyctalus_noctula_roost"
    
    def test_parts_with_whitespace(self):
        """Test with parts containing whitespace."""
        assert get_model_id(["Myotis daubentonii", "In flight"]) == "myotis_daubentonii_in_flight"
        assert get_model_id(["  Test  ", "  Part  "]) == "test_part"
    
    def test_parts_with_mixed_case(self):
        """Test with parts containing mixed case."""
        assert get_model_id(["Myotis Daubentonii", "In Flight"]) == "myotis_daubentonii_in_flight"
    
    def test_empty_list(self):
        """Test with empty list."""
        assert get_model_id([]) == ""

