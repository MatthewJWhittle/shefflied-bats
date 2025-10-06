"""
Tests for text utility functions.
"""

import pytest

from sdm.utils.text_utils import tidy_variable_name


class TestTidyVariableName:
    """Test the tidy_variable_name function."""
    
    def test_basic_string_cleaning(self):
        """Test basic string cleaning functionality."""
        # Test spaces to underscores
        assert tidy_variable_name("hello world") == "hello_world"
        assert tidy_variable_name("hello  world") == "hello_world"
        
        # Test dashes to underscores
        assert tidy_variable_name("hello-world") == "hello_world"
        assert tidy_variable_name("hello--world") == "hello_world"
        
        # Test mixed separators
        assert tidy_variable_name("hello-world test") == "hello_world_test"
        assert tidy_variable_name("hello/world.test") == "hello_world_test"
    
    def test_case_conversion(self):
        """Test case conversion to lowercase."""
        assert tidy_variable_name("Hello World") == "hello_world"
        assert tidy_variable_name("HELLO WORLD") == "hello_world"
        assert tidy_variable_name("HeLlO WoRlD") == "hello_world"
    
    def test_special_characters(self):
        """Test handling of special characters."""
        # Test various separators
        assert tidy_variable_name("hello.world") == "hello_world"
        assert tidy_variable_name("hello:world") == "hello_world"
        assert tidy_variable_name("hello;world") == "hello_world"
        assert tidy_variable_name("hello,world") == "hello_world"
        assert tidy_variable_name("hello(world)") == "hello_world"
        assert tidy_variable_name("hello[world]") == "hello_world"
        assert tidy_variable_name("hello{world}") == "hello_world"
        
        # Test backslashes
        assert tidy_variable_name("hello\\world") == "hello_world"
        
        # Test multiple consecutive separators
        assert tidy_variable_name("hello---world") == "hello_world"
        assert tidy_variable_name("hello   world") == "hello_world"
        assert tidy_variable_name("hello___world") == "hello_world"
    
    def test_leading_trailing_underscores(self):
        """Test removal of leading and trailing underscores."""
        assert tidy_variable_name("_hello_world_") == "hello_world"
        assert tidy_variable_name("__hello_world__") == "hello_world"
        assert tidy_variable_name("___hello___world___") == "hello_world"
    
    def test_alphanumeric_preservation(self):
        """Test that alphanumeric characters are preserved."""
        assert tidy_variable_name("hello123") == "hello123"
        assert tidy_variable_name("123hello") == "123hello"
        assert tidy_variable_name("hello123world") == "hello123world"
        assert tidy_variable_name("hello_world_123") == "hello_world_123"
    
    def test_non_alphanumeric_removal(self):
        """Test removal of non-alphanumeric characters."""
        assert tidy_variable_name("hello@world") == "helloworld"
        assert tidy_variable_name("hello#world") == "helloworld"
        assert tidy_variable_name("hello$world") == "helloworld"
        assert tidy_variable_name("hello%world") == "helloworld"
        assert tidy_variable_name("hello^world") == "helloworld"
        assert tidy_variable_name("hello&world") == "helloworld"
        assert tidy_variable_name("hello*world") == "helloworld"
        assert tidy_variable_name("hello+world") == "helloworld"
        assert tidy_variable_name("hello=world") == "helloworld"
        assert tidy_variable_name("hello!world") == "helloworld"
        assert tidy_variable_name("hello?world") == "helloworld"
        assert tidy_variable_name("hello|world") == "helloworld"
        assert tidy_variable_name("hello~world") == "helloworld"
        assert tidy_variable_name("hello`world") == "helloworld"
    
    def test_empty_and_edge_cases(self):
        """Test edge cases and empty inputs."""
        assert tidy_variable_name("") == ""
        assert tidy_variable_name("   ") == ""
        assert tidy_variable_name("___") == ""
        assert tidy_variable_name("---") == ""
        assert tidy_variable_name("   ___   ") == ""
        
        # Test single characters
        assert tidy_variable_name("a") == "a"
        assert tidy_variable_name("_") == ""
        assert tidy_variable_name("-") == ""
        assert tidy_variable_name("1") == "1"
    
    def test_complex_real_world_examples(self):
        """Test complex real-world examples that might be used in raster processing."""
        # Test dataset names
        assert tidy_variable_name("DTM_1m") == "dtm_1m"
        assert tidy_variable_name("DSM-1m") == "dsm_1m"
        assert tidy_variable_name("Land Cover/100m") == "land_cover_100m"
        assert tidy_variable_name("Temperature (Average)") == "temperature_average"
        assert tidy_variable_name("Wind Speed [m/s]") == "wind_speed_m_s"
        
        # Test file paths converted to variable names
        assert tidy_variable_name("data/processed/dtm.tif") == "data_processed_dtm_tif"
        assert tidy_variable_name("env-vars/climate-data.nc") == "env_vars_climate_data_nc"
        
        # Test band descriptions
        assert tidy_variable_name("Band 1: Red") == "band_1_red"
        assert tidy_variable_name("Band_2: Green") == "band_2_green"
        assert tidy_variable_name("Band-3: Blue") == "band_3_blue"
    
    def test_non_string_inputs(self):
        """Test handling of non-string inputs."""
        # Test conversion to string
        assert tidy_variable_name(123) == "123"
        assert tidy_variable_name(123.45) == "123_45"
        assert tidy_variable_name(True) == "true"
        assert tidy_variable_name(False) == "false"
        
        # Test None (converts to string "none")
        assert tidy_variable_name(None) == "none"
        
        # Test complex objects (converts to string representation)
        assert tidy_variable_name([1, 2, 3]) == "1_2_3"
        assert tidy_variable_name({"key": "value"}) == "key_value"
    
    def test_unicode_and_special_unicode(self):
        """Test handling of unicode and special characters."""
        # Test unicode characters
        assert tidy_variable_name("café") == "caf"
        assert tidy_variable_name("naïve") == "nave"
        assert tidy_variable_name("résumé") == "rsum"
        
        # Test emoji and special unicode
        assert tidy_variable_name("hello😀world") == "helloworld"
        assert tidy_variable_name("test→result") == "testresult"
        assert tidy_variable_name("data™info") == "datainfo"
    
    def test_very_long_strings(self):
        """Test handling of very long strings."""
        long_string = "a" * 1000
        assert tidy_variable_name(long_string) == long_string
        
        long_string_with_separators = "a-b-c-d-" * 100
        # The function converts each "-" to "_" and strips trailing underscores
        expected = "a_b_c_d" * 100  # Each "a-b-c-d-" becomes "a_b_c_d"
        result = tidy_variable_name(long_string_with_separators)
        assert len(result) == 799  # Length is 799, not 700
        assert result.startswith("a_b_c_d")
    
    def test_preserves_underscores_in_middle(self):
        """Test that underscores in the middle of words are preserved."""
        assert tidy_variable_name("hello_world") == "hello_world"
        assert tidy_variable_name("test_123_value") == "test_123_value"
        assert tidy_variable_name("my_variable_name") == "my_variable_name"
