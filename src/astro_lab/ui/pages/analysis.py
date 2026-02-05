"""
Analysis Page
============

General analysis page (simplified - main functionality in cosmic_web.py).
"""

import marimo as mo


def create_page(app_state=None):
    """Create the general analysis page."""

    # Check if data is loaded
    if (
        app_state is None
        or not hasattr(app_state, "loaded_data")
        or app_state.loaded_data is None
    ):
        return mo.vstack(
            [
                mo.md("## 🔬 Analysis"),
                mo.callout(
                    "Please load data first in the **Data** tab.", kind="warning"
                ),
                mo.md(
                    "After loading data, use the **Cosmic Web** tab for detailed analysis."
                ),
            ]
        )

    # Simple analysis info
    data_info = mo.vstack(
        [
            mo.md("## 🔬 Analysis"),
            mo.md("Data is loaded and ready for analysis."),
            mo.hstack(
                [
                    mo.stat(
                        "Objects",
                        f"{app_state.n_objects:,}"
                        if hasattr(app_state, "n_objects")
                        else "Unknown",
                    ),
                    mo.stat("Status", "Ready"),
                ]
            ),
            mo.md("### Available Analysis Types"),
            mo.md(
                "- **Cosmic Web Analysis**: Go to the Cosmic Web tab for detailed structure analysis"
            ),
            mo.md(
                "- **Visualization**: Go to the Visualization tab for interactive plots"
            ),
        ]
    )

    return data_info
