"""Test script to verify config integration fixes."""

import sys

sys.path.insert(0, "src")

from astro_lab.config import (
    get_combined_config,
    get_config,
    get_survey_config,
    get_task_config,
)
from astro_lab.config_validator import ConfigValidator


def test_config_loading():
    """Test that all configs load correctly."""
    print("Testing config loading...")

    # Test basic config loading
    config = get_config()
    print(f"✓ Loaded {len(config)} config sections")

    # Test survey config
    try:
        gaia_config = get_survey_config("gaia")
        print(f"✓ Gaia config loaded with {len(gaia_config)} fields")
        if "recommended_model" in gaia_config:
            print(
                f"  ✓ Recommended model: {gaia_config['recommended_model']['conv_type']}"
            )
    except Exception as e:
        print(f"✗ Failed to load Gaia config: {e}")

    # Test combined config
    try:
        combined = get_combined_config("gaia", "node_classification")
        print(f"✓ Combined config has {len(combined)} fields")
        print(f"  - conv_type: {combined.get('conv_type', 'not set')}")
        print(f"  - batch_size: {combined.get('batch_size', 'not set')}")
        print(f"  - max_epochs: {combined.get('max_epochs', 'not set')}")
    except Exception as e:
        print(f"✗ Failed to get combined config: {e}")

    # Test task config
    try:
        task_config = get_task_config("node_classification")
        print(f"✓ Task config: {task_config}")
    except Exception as e:
        print(f"✗ Failed to get task config: {e}")

    print("\n" + "=" * 60 + "\n")


def test_config_validation():
    """Test config validation."""
    print("Running config validation...")
    validator = ConfigValidator()
    is_valid, issues = validator.validate_all()

    if is_valid:
        print("✅ All configs are valid!")
    else:
        print(f"❌ Found {len(issues)} validation issues:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")

    print("\n" + "=" * 60 + "\n")


def test_survey_recommendations():
    """Test survey-specific model recommendations."""
    print("Testing survey model recommendations...")

    surveys = ["gaia", "sdss", "nsa", "exoplanet"]
    for survey in surveys:
        try:
            config = get_survey_config(survey)
            if "recommended_model" in config:
                rec = config["recommended_model"]
                print(
                    f"✓ {survey}: {rec['conv_type']} with {rec.get('num_layers', '?')} layers"
                )
            else:
                print(f"⚠ {survey}: No model recommendation")
        except Exception as e:
            print(f"✗ {survey}: Failed to load - {e}")

    print("\n" + "=" * 60 + "\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AstroLab Config Integration Test")
    print("=" * 60 + "\n")

    test_config_loading()
    test_config_validation()
    test_survey_recommendations()

    print("✅ Config integration test completed!")


if __name__ == "__main__":
    main()
