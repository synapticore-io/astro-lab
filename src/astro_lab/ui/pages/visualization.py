"""
Visualization Page
================

Interactive visualization page using real AstroLab widgets.
"""

import marimo as mo

from astro_lab.ui.components.viz import (
    create_cosmic_web_viz,
    create_plotly_viz,
    create_visualizer,
)


def create_page(app_state=None):
    """Create the visualization page."""

    # Check if data is loaded
    if (
        app_state is None
        or not hasattr(app_state, "loaded_data")
        or app_state.loaded_data is None
    ):
        return mo.vstack(
            [
                mo.md("## 🎨 Visualization"),
                mo.callout(
                    "Please load data first in the **Data** tab.", kind="warning"
                ),
            ]
        )

    # Visualization controls
    viz_ui, viz_config, viz_status = create_visualizer()

    # Create visualization based on config
    visualization = mo.md("")

    if viz_config and viz_config.get("ready"):
        try:
            backend = viz_config["backend"]
            node_size = viz_config["node_size"]
            color_by = viz_config["color_by"]

            if backend == "plotly":
                visualization = create_plotly_viz(
                    data=app_state.loaded_data, color_by=color_by, node_size=node_size
                )
            elif backend == "cosmograph":
                # Use analysis results if available
                analysis_results = getattr(app_state, "analysis_result", None)
                visualization = create_cosmic_web_viz(
                    data=app_state.loaded_data, analysis_results=analysis_results
                )
        except Exception as e:
            visualization = mo.callout(f"Visualization error: {str(e)}", kind="danger")

    # Data summary
    data_summary = mo.md("")
    if hasattr(app_state, "loaded_data") and app_state.loaded_data is not None:
        n_objects = len(app_state.loaded_data)
        has_3d = all(col in app_state.loaded_data.columns for col in ["x", "y", "z"])

        data_summary = mo.vstack(
            [
                mo.md("### 📊 Data Summary"),
                mo.hstack(
                    [
                        mo.stat("Objects", f"{n_objects:,}"),
                        mo.stat("3D Coords", "✅" if has_3d else "❌"),
                        mo.stat(
                            "Analysis",
                            "✅" if hasattr(app_state, "analysis_result") else "❌",
                        ),
                    ]
                ),
            ]
        )

    # Layout
    return mo.vstack(
        [
            mo.md("## 🎨 Interactive Visualization"),
            data_summary,
            mo.hstack(
                [
                    # Left: Controls
                    mo.vstack(
                        [
                            viz_ui,
                            viz_status,
                        ],
                        align="stretch",
                    ),
                    # Right: Visualization
                    mo.vstack([visualization], align="stretch"),
                ],
                widths=[1, 2],
            ),
        ]
    )
