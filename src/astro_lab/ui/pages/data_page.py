"""
Data Page - Using existing AstroLab data loading components
Following Marimo 2025 best practices with existing components
"""

import marimo as mo

from ..components import state

# Use existing AstroLab components with relative imports
from ..components.data_loader import create_data_loader
from ..components.viz import create_visualizer


def create_data_page():
    """Create data page using existing AstroLab components"""

    # Use existing data loader component
    data_ui, status, preview, loaded_data = create_data_loader()

    # Update global state if data was loaded
    if loaded_data is not None:
        current_state = state.get_state()
        current_state["loaded_data"] = loaded_data
        state.set_state(current_state)

    # Create visualization if data is available
    viz_component = None
    if loaded_data is not None:
        viz_component = create_data_visualization_section(loaded_data)

    return mo.vstack(
        [
            mo.md("# 📡 Data Loading & Visualization"),
            mo.md(
                "Load and process real astronomical survey data using AstroLab's data pipeline."
            ),
            mo.md("---"),
            # Data Loading Section (using existing component)
            data_ui,
            status,
            mo.md("---"),
            # Data Preview
            mo.md("## 📊 Data Preview"),
            preview,
            mo.md("---"),
            # Visualization Section
            mo.md("## 🎨 Data Visualization"),
            viz_component
            if viz_component
            else mo.md("⏳ **Load data first to enable visualization**"),
        ]
    )


def create_data_visualization_section(loaded_data):
    """Create visualization section for loaded data"""

    if loaded_data is None:
        return mo.md("⏳ **No data available for visualization**")

    try:
        # Check if data has spatial coordinates
        has_coordinates = check_for_coordinates(loaded_data)

        if not has_coordinates:
            return mo.md(
                "⚠️ **Loaded data has no spatial coordinates for visualization**"
            )

        # Use existing viz component
        viz_ui, viz_config, viz_status = create_visualizer()

        return mo.vstack(
            [
                mo.md("✅ **Spatial coordinates detected!** Ready for visualization."),
                viz_ui,
                viz_status,
                mo.md("💡 Use the visualization controls above to create plots."),
            ]
        )

    except Exception as e:
        return mo.md("❌ **Visualization error:** " + str(e))


def check_for_coordinates(data):
    """Check if data contains spatial coordinates"""

    try:
        # Check for different coordinate types
        if hasattr(data, "columns"):
            # Polars DataFrame
            cols = data.columns
        elif hasattr(data, "keys"):
            # Dictionary-like
            cols = list(data.keys())
        else:
            # Try to get columns from dataframe attribute
            cols = getattr(data, "columns", [])

        # Look for common coordinate column names
        coord_cols = [
            "ra",
            "dec",
            "x",
            "y",
            "z",
            "coordinates",
            "galactic_l",
            "galactic_b",
        ]
        return any(col in cols for col in coord_cols)

    except Exception:
        return False


def create_data_info_display():
    """Display information about currently loaded data"""

    current_state = state.get_state()
    loaded_data = current_state.get("loaded_data")

    if loaded_data is None:
        return mo.md("📭 **No data currently loaded**")

    try:
        # Get basic info about the data
        data_type = type(loaded_data).__name__

        # Try to get size info
        try:
            if hasattr(loaded_data, "__len__"):
                size = len(loaded_data)
            elif hasattr(loaded_data, "shape"):
                size = loaded_data.shape[0]
            else:
                size = "Unknown"
        except Exception:
            size = "Unknown"

        # Try to get feature info
        try:
            if hasattr(loaded_data, "columns"):
                n_features = len(loaded_data.columns)
                sample_cols = list(loaded_data.columns)[:5]
            elif hasattr(loaded_data, "num_features"):
                n_features = loaded_data.num_features
                sample_cols = ["Feature extraction needed"]
            else:
                n_features = "Unknown"
                sample_cols = []
        except Exception:
            n_features = "Unknown"
            sample_cols = []

        info_md = """
        ## 📊 Currently Loaded Data
        
        **Type:** {}
        **Size:** {} objects
        **Features:** {}
        **Sample Columns:** {}
        **Spatial Coordinates:** {}
        """.format(
            data_type,
            size,
            n_features,
            ", ".join(sample_cols),
            "✅" if check_for_coordinates(loaded_data) else "❌",
        )

        return mo.md(info_md)

    except Exception as e:
        return mo.md("⚠️ **Could not read data info:** " + str(e))
