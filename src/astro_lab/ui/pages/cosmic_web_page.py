"""
Cosmic Web Page - Using existing AstroLab analyzer components
Following Marimo 2025 best practices with existing components
"""

import marimo as mo

from ..components import state

# Use existing AstroLab components with relative imports
from ..components.analyzer import create_analyzer, run_cosmic_web_analysis
from ..components.viz import create_viz_component


def create_cosmic_web_page():
    """Create cosmic web page using existing AstroLab components"""

    # Check if data is loaded
    current_state = state.get_state()
    loaded_data = current_state.get("loaded_data")

    if loaded_data is None:
        return mo.vstack(
            [
                mo.md("# 🌌 Cosmic Web Analysis"),
                mo.md("⚠️ **No data loaded!**"),
                mo.md("Go to the **Data** tab and load survey data first."),
                mo.md("Cosmic web analysis requires 3D spatial coordinates."),
            ]
        )

    # Use existing analyzer component
    analyzer_ui, analyzer_status, analyzer_config = create_analyzer()

    # Run analysis if configured
    analysis_results = None
    analysis_status = mo.md("")

    if analyzer_config and analyzer_config.get("configured"):
        try:
            analysis_status = mo.md(
                "🔄 **Running cosmic web analysis...** Please wait."
            )

            analysis_results = run_cosmic_web_analysis(
                data=loaded_data,
                scales=analyzer_config["scales"],
                min_samples=analyzer_config["min_samples"],
                method=analyzer_config["method"],
            )

            if analysis_results.get("error"):
                analysis_status = mo.md(
                    "❌ **Analysis failed:** " + analysis_results["error"]
                )
            elif analysis_results.get("analysis_complete"):
                analysis_status = mo.md(
                    "✅ **Cosmic web analysis completed successfully!**"
                )
            else:
                analysis_status = mo.md("⚠️ **Analysis completed with warnings**")

        except Exception as e:
            analysis_status = mo.md("❌ **Analysis error:** " + str(e))

    return mo.vstack(
        [
            mo.md("# 🌌 Cosmic Web Analysis"),
            mo.md(
                "Analyze cosmic structures using AstroLab's cosmic web detection algorithms."
            ),
            mo.md("---"),
            # Data Status
            mo.md("## 📊 Analysis Data Status"),
            create_analysis_data_status(loaded_data),
            mo.md("---"),
            # Analysis Configuration (using existing component)
            analyzer_ui,
            analyzer_status,
            mo.md("---"),
            # Analysis Results
            mo.md("## 📈 Analysis Results"),
            analysis_status,
            create_analysis_results_display(analysis_results),
            mo.md("---"),
            # Visualization
            mo.md("## 🎨 Results Visualization"),
            create_analysis_visualization_section(analysis_results),
        ]
    )


def create_analysis_data_status(loaded_data):
    """Display status of data for cosmic web analysis"""

    if loaded_data is None:
        return mo.md("❌ **No data available for analysis**")

    try:
        # Check data suitability for cosmic web analysis
        has_spatial = check_spatial_coordinates(loaded_data)

        if not has_spatial:
            return mo.md(
                "⚠️ **Loaded data lacks spatial coordinates needed for cosmic web analysis**"
            )

        # Get data size
        try:
            if hasattr(loaded_data, "__len__"):
                size = len(loaded_data)
            elif hasattr(loaded_data, "shape"):
                size = loaded_data.shape[0]
            else:
                size = "Unknown"
        except Exception:
            size = "Unknown"

        coord_type = get_coordinate_type(loaded_data)

        status_md = """
        ✅ **Data ready for cosmic web analysis!**
        
        - **Objects:** {}
        - **Coordinate Type:** {}
        - **Spatial Coordinates:** ✅ Available
        - **Analysis Ready:** ✅ Yes
        """.format(size, coord_type)

        return mo.md(status_md)

    except Exception as e:
        return mo.md("⚠️ **Could not assess data:** " + str(e))


def check_spatial_coordinates(data):
    """Check if data has spatial coordinates for cosmic web analysis"""

    try:
        if hasattr(data, "columns"):
            cols = data.columns
        elif hasattr(data, "keys"):
            cols = list(data.keys())
        else:
            return False

        # Check for 3D Cartesian coordinates
        has_cartesian = all(col in cols for col in ["x", "y", "z"])

        # Check for spherical coordinates with distance
        has_spherical = all(col in cols for col in ["ra", "dec"]) and any(
            col in cols for col in ["distance_pc", "distance", "parallax"]
        )

        return has_cartesian or has_spherical

    except Exception:
        return False


def get_coordinate_type(data):
    """Determine the type of coordinates in the data"""

    try:
        if hasattr(data, "columns"):
            cols = data.columns
        elif hasattr(data, "keys"):
            cols = list(data.keys())
        else:
            return "Unknown"

        if all(col in cols for col in ["x", "y", "z"]):
            return "3D Cartesian (x, y, z)"
        elif all(col in cols for col in ["ra", "dec"]):
            if any(col in cols for col in ["distance_pc", "distance"]):
                return "Spherical (RA, DEC, Distance)"
            else:
                return "Sky Coordinates (RA, DEC)"
        else:
            return "Other/Mixed"

    except Exception:
        return "Unknown"


def create_analysis_results_display(results):
    """Display cosmic web analysis results"""

    if results is None:
        return mo.md("⏳ **Configure and run analysis to see results**")

    if results.get("error"):
        return mo.md("❌ **Analysis Error:** " + results["error"])

    if not results.get("analysis_complete"):
        return mo.md("⚠️ **Analysis incomplete or failed**")

    # Display results summary
    try:
        n_objects = results.get("n_objects", 0)
        scales = results.get("scales", [])
        method = results.get("method", "Unknown")
        total_structures = results.get("total_structures", 0)

        # Get cosmic web structure counts if available
        cosmic_web = results.get("cosmic_web", {})

        structures_info = ""
        if cosmic_web:
            filaments = len(cosmic_web.get("filaments", []))
            clusters = len(cosmic_web.get("clusters", []))
            voids = len(cosmic_web.get("voids", []))
            walls = len(cosmic_web.get("walls", []))

            structures_info = """
            **Detected Structures:**
            - Filaments: {}
            - Clusters: {}
            - Voids: {}
            - Walls: {}
            """.format(filaments, clusters, voids, walls)

        results_md = """
        ## ✅ Cosmic Web Analysis Complete
        
        **Objects Analyzed:** {:,}
        **Scales Used:** {} pc
        **Method:** {}
        **Total Structures:** {}
        
        {}
        
        💡 Higher structure counts indicate richer cosmic web features.
        """.format(
            n_objects,
            ", ".join(map(str, scales)),
            method.upper(),
            total_structures,
            structures_info,
        )

        return mo.md(results_md)

    except Exception as e:
        return mo.md("⚠️ **Could not display results:** " + str(e))


def create_analysis_visualization_section(results):
    """Create visualization section for analysis results"""

    if results is None or not results.get("analysis_complete"):
        return mo.md("⏳ **Complete analysis first to enable visualization**")

    # Use existing viz component for results
    viz_ui = create_viz_component()

    return mo.vstack(
        [
            mo.md("✅ **Analysis complete!** Use visualization controls below."),
            viz_ui,
            mo.md("💡 Visualizations will show detected cosmic web structures."),
        ]
    )
