import pandas as pd
import pytest
import numpy as np

# Assuming AggregateFeatures is accessible (e.g., copied to a 'utils' module or imported directly)
# For simplicity, we define a small utility function here to test.


def check_for_nan(df, column):
    """Utility function to check if a column contains NaN values."""
    return df[column].isnull().any()

# --- Mock Data ---


@pytest.fixture
def sample_data():
    """Fixture to provide a clean DataFrame for testing."""
    data = {
        'CustomerId': ['A', 'B', 'A', 'C', 'B'],
        'Amount': [100, 200, 150, 50, 250],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'CountryCode': ['UG', 'KE', 'UG', 'UG', 'KE']
    }
    return pd.DataFrame(data)

# --- Unit Tests ---


def test_dataframe_shape_and_columns(sample_data):
    """Test 1: Verify the shape and expected columns of the raw fixture data."""
    assert sample_data.shape == (5, 4)
    assert 'Amount' in sample_data.columns
    assert 'CustomerId' in sample_data.columns


def test_nan_check_utility_function():
    """Test 2: Test the behavior of a simple utility function."""
    test_df = pd.DataFrame({'col1': [1, 2, np.nan, 4]})

    # Test column with NaN
    assert check_for_nan(test_df, 'col1') == True

    # Test column without NaN
    test_df['col2'] = [5, 6, 7, 8]
    assert check_for_nan(test_df, 'col2') == False

# Instructions for user: To run tests, navigate to the project root and execute:
# 'pytest'
