# Refactoring Guide for Sheffield Bats SDM Repository

This guide provides clear, step-by-step instructions for refactoring scripts from the `old_code/` directory to the new modular `sdm/` structure. Follow these instructions to ensure refactored scripts maintain identical functionality while improving code organization and maintainability.

---

## **📁 Repository Structure**

### **Key Directories:**
- **`old_code/`** - Original scripts that need refactoring
- **`sdm/`** - New modular package structure
- **`tests/`** - Test files for both old and new implementations

### **New Package Structure:**
```
sdm/
├── commands/data_preparation/environmental/  # Main script entry points
├── data/                                     # Data processing modules
├── utils/                                    # Utility functions
├── raster/                                   # Raster processing utilities
└── models/                                   # Modeling functionality
```

### **Test Structure:**
```
tests/
├── test_[old_script_name].py              # Tests for individual functions
└── test_[new_script_name].py              # Integration tests for workflows
```

---

## **🔄 Step-by-Step Refactoring Process**

### **Step 1: Initial Analysis**
1. **Locate the old script** in `old_code/`
2. **Find the corresponding new script** in `sdm/commands/data_preparation/environmental/`
3. **Read both scripts** to understand the overall workflow
4. **Identify core functions** and their responsibilities in the old script
5. **Map old functions** to new modular structure

### **Step 2: Import and Dependency Verification**
1. **Check all imports** in the new script
2. **Verify imported modules exist** in the `sdm` package
3. **Examine function signatures** to ensure they match usage
4. **Test imports** by running: `python -c "from sdm.module import function"`
5. **Fix any import errors** or missing dependencies

### **Step 3: Function Logic Alignment**
1. **Compare function parameters** between old and new implementations
2. **Verify CRS handling** - ensure consistent coordinate reference system usage
3. **Check spatial operations** - resolution, coarsening, reprojection logic
4. **Validate data transformations** - category mappings, aggregations, calculations
5. **Confirm default parameters** match original script values
6. **Test error handling** - ensure appropriate exceptions are raised

### **Step 4: Code Quality and Style**
1. **Run linting**: `python -m flake8 sdm/` and `python -m flake8 tests/`
2. **Fix linting issues**:
   - Remove unused imports
   - Use lazy % formatting in logging: `logging.info("Message %s", variable)`
   - Fix mutable default arguments: `param=None` then handle inside function
   - Remove unused variables
3. **Ensure type hints** are present and correct
4. **Verify documentation** is clear and complete

### **Step 5: Test Implementation**
1. **Update existing tests** to use new import paths
2. **Create unit tests** for individual functions in `sdm` modules
3. **Create integration tests** for complete workflows
4. **Follow test best practices** (see Testing Guidelines below)
5. **Run tests**: `python -m pytest tests/test_[script_name].py -v`

---

## **🧪 Testing Guidelines**

### **Test Design Principles:**
- **Test real functionality** - avoid over-mocking
- **Use minimal mocking** - only for external dependencies (network, file I/O)
- **Keep tests simple** - focus on core functionality
- **Use synthetic data** - avoid large dataset dependencies
- **Test error conditions** - missing files, invalid inputs

### **Test Structure:**
```python
# Good test example
def test_function_with_real_data(tmp_path):
    """Test function with actual data processing."""
    # Create test data
    test_data = create_test_data()
    
    # Run function
    result = function_under_test(test_data, tmp_path)
    
    # Verify results
    assert result is not None
    assert expected_condition
```

### **What NOT to do:**
```python
# Avoid over-mocking
@patch('module.function1')
@patch('module.function2')
@patch('module.function3')
@patch('module.function4')
@patch('module.function5')
def test_over_mocked():
    # This tests mocked behavior, not real functionality
```

### **Import Organization:**
```python
# Good - all imports at top
import pytest
import numpy as np
from pathlib import Path
from sdm.module import function

# Bad - imports inside functions
def test_function():
    from sdm.module import function  # Don't do this
```

---

## **🔍 Common Issues and Solutions**

### **Function Signature Mismatches**
- **Problem**: Parameters don't match between old and new code
- **Solution**: Update function signatures or parameter handling
- **Check**: Verify all function calls match their definitions

### **CRS and Spatial Data Issues**
- **Problem**: Coordinate reference system handling differs
- **Solution**: Ensure consistent CRS usage throughout pipeline
- **Check**: Verify reprojection and spatial operations use correct CRS

### **Resolution and Coarsening Problems**
- **Problem**: Pixel resolution calculations incorrect
- **Solution**: Double-check resolution extraction and coarsening factors
- **Check**: `rio.resolution()` is a method, not property

### **Default Parameter Issues**
- **Problem**: Mutable default arguments cause unexpected behavior
- **Solution**: Use `None` defaults and handle inside function
- **Example**: `param=None` then `if param is None: param = []`

### **Output Format Mismatches**
- **Problem**: New script produces different output format
- **Solution**: Compare with existing output files to verify expected format
- **Check**: File extensions, data structure, band names, etc.

---

## **✅ Verification Checklist**

### **Before Starting:**
- [ ] Located old script in `old_code/`
- [ ] Found corresponding new script in `sdm/`
- [ ] Read both scripts to understand workflow

### **During Refactoring:**
- [ ] All imports resolve correctly
- [ ] Function signatures match usage
- [ ] Default parameters handled correctly
- [ ] CRS operations consistent
- [ ] Spatial calculations correct
- [ ] Error handling appropriate

### **Code Quality:**
- [ ] No linting errors
- [ ] No unused imports or variables
- [ ] Lazy logging format used
- [ ] Type hints present
- [ ] Documentation clear

### **Testing:**
- [ ] Existing tests updated and passing
- [ ] New unit tests created
- [ ] Integration tests implemented
- [ ] Error conditions tested
- [ ] All tests pass: `pytest tests/ -v`

### **Final Verification:**
- [ ] Script produces identical output to original
- [ ] All expected files created
- [ ] File formats and structure correct
- [ ] Performance acceptable
- [ ] Documentation updated

---

## **🛠️ Useful Commands**

### **Testing Commands:**
```bash
# Run specific test file
python -m pytest tests/test_[script_name].py -v

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=sdm

# Run linting
python -m flake8 sdm/ tests/
```

### **Import Testing:**
```bash
# Test imports work
python -c "from sdm.module import function; print('Import successful')"

# Test function signatures
python -c "import inspect; from sdm.module import function; print(inspect.signature(function))"
```

### **File Verification:**
```bash
# Check file structure
find sdm/ -name "*.py" | head -10

# Compare file sizes
ls -la old_code/script.py sdm/commands/script.py
```

---

## **📚 Examples and Patterns**

### **Good Function Signature:**
```python
def process_data(
    input_data: xr.DataArray,
    output_path: Path,
    parameter: str = None
) -> Path:
    """Process data with clear parameters."""
    if parameter is None:
        parameter = "default_value"
    # Function implementation
    return output_path
```

### **Good Test Structure:**
```python
def test_process_data(tmp_path):
    """Test data processing with real functionality."""
    # Create test data
    test_data = create_test_data()
    
    # Run function
    result = process_data(test_data, tmp_path / "output.tif")
    
    # Verify results
    assert result.exists()
    assert result.suffix == ".tif"
```

### **Error Handling Pattern:**
```python
try:
    result = risky_operation()
except FileNotFoundError:
    logging.error("File not found: %s", file_path)
    raise
except Exception as e:
    logging.error("Unexpected error: %s", e)
    raise
```

---

## **🎯 Success Criteria**

A refactored script is considered successful when:

1. **Functionality Preserved**: Produces identical output to original script
2. **Code Quality**: Passes all linting checks with no warnings
3. **Test Coverage**: Has comprehensive tests covering core functionality
4. **Performance**: Runs efficiently without significant slowdown
5. **Maintainability**: Code is clear, well-documented, and follows best practices
6. **Integration**: Works seamlessly with the new modular structure

---

*This guide is updated based on lessons learned from refactoring CEH land cover and climate data processing scripts. Continue updating this guide as new patterns and solutions are discovered.*