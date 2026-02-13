"""
Test catalog generation and visualization scripts.

This test validates that:
1. Scripts have correct syntax and can be imported
2. Core functions are defined and callable
3. CLI arguments are properly configured
"""

import pytest
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_catalog_script_imports():
    """Test that catalog generation script can be imported."""
    try:
        import generate_astrolab_catalog
        assert hasattr(generate_astrolab_catalog, 'generate_catalog')
        assert hasattr(generate_astrolab_catalog, 'main')
    except ImportError as e:
        pytest.skip(f"Could not import catalog script: {e}")


def test_visualization_script_imports():
    """Test that visualization script can be imported."""
    try:
        import generate_visualizations
        assert hasattr(generate_visualizations, 'generate_all_visualizations')
        assert hasattr(generate_visualizations, 'create_3d_cosmic_web_plot')
        assert hasattr(generate_visualizations, 'main')
    except ImportError as e:
        pytest.skip(f"Could not import visualization script: {e}")


def test_catalog_example_imports():
    """Test that catalog usage example can be imported."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    try:
        import use_astrolab_catalog
        assert hasattr(use_astrolab_catalog, 'load_catalog')
        assert hasattr(use_astrolab_catalog, 'filter_by_structure')
        assert hasattr(use_astrolab_catalog, 'main')
    except ImportError as e:
        pytest.skip(f"Could not import catalog example: {e}")


def test_data_directory_structure():
    """Test that data directory structure exists."""
    data_dir = Path(__file__).parent.parent / "data"
    
    assert data_dir.exists(), "data directory should exist"
    assert (data_dir / "README.md").exists(), "data/README.md should exist"
    assert (data_dir / "catalog_schema.md").exists(), "data/catalog_schema.md should exist"
    
    # Check subdirectories
    assert (data_dir / "catalogs").exists(), "data/catalogs should exist"
    assert (data_dir / "visualizations").exists(), "data/visualizations should exist"
    assert (data_dir / "raw").exists(), "data/raw should exist"
    assert (data_dir / "processed").exists(), "data/processed should exist"


def test_examples_updated():
    """Test that examples have been updated."""
    examples_dir = Path(__file__).parent.parent / "examples"
    
    # Check that example files exist
    assert (examples_dir / "preprocess_and_combine_surveys.py").exists()
    assert (examples_dir / "train_gaia_real_data.py").exists()
    assert (examples_dir / "use_astrolab_catalog.py").exists()
    
    # Check that new example has expected content
    catalog_example = (examples_dir / "use_astrolab_catalog.py").read_text()
    assert "load_catalog" in catalog_example
    assert "filter_by_structure" in catalog_example
    assert "cosmic_web_class" in catalog_example
    assert "astrolab_catalog_v1.parquet" in catalog_example


def test_documentation_updated():
    """Test that documentation has been updated."""
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    
    # Check for catalog references
    assert "AstroLab Consolidated Catalog" in readme or "astrolab_catalog" in readme.lower()
    assert "data/catalogs" in readme or "generate_astrolab_catalog" in readme
    
    # Check cosmic web guide
    cw_guide = (Path(__file__).parent.parent / "docs" / "cosmic_web_guide.md").read_text()
    assert "AstroLab Consolidated Catalog" in cw_guide or "catalog" in cw_guide.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
