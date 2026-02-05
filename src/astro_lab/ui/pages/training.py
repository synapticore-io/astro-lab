"""
Training Page
============

Simple model training interface.
"""

import marimo as mo


def create_page(app_state=None):
    """Create the training page."""

    # Check if data is loaded
    if app_state is None or app_state.loaded_data is None:
        return mo.vstack(
            [
                mo.md("## 🏋️ Model Training"),
                mo.callout(
                    "Please load data first before training models!", kind="warning"
                ),
                mo.ui.button("Go to Data Tab", kind="primary"),
            ]
        )

    # Model selection
    model_selector = mo.ui.dropdown(
        options={
            "AstroGraphGNN": "Graph Neural Network for astronomical data",
            "AstroNodeGNN": "Node classification model",
            "AstroPointNet": "Point cloud processing model",
        },
        value="AstroGraphGNN",
        label="Select Model",
    )

    # Training parameters
    epochs = mo.ui.slider(10, 200, 50, step=10, label="Epochs")
    batch_size = mo.ui.dropdown(options=[16, 32, 64, 128], value=32, label="Batch Size")
    learning_rate = mo.ui.number(0.0001, 0.1, 0.001, step=0.0001, label="Learning Rate")

    # Training controls
    start_btn = mo.ui.button("🚀 Start Training", kind="success", full_width=True)
    stop_btn = mo.ui.button("⏹️ Stop", kind="danger", disabled=True)

    # Progress display
    progress = mo.md("")
    status = mo.md("")

    if start_btn.value:
        # Simulate training start
        status = mo.callout(
            f"✅ Training {model_selector.value} for {epochs.value} epochs", kind="info"
        )

        # Update app state
        if app_state:
            app_state.training_status = "Training"
            app_state.total_epochs = epochs.value

    # Results section
    results = mo.md("")
    if app_state and hasattr(app_state, "training_result"):
        results = mo.vstack(
            [
                mo.md("### 📊 Training Results"),
                mo.stat("Final Loss", "0.0234"),
                mo.stat("Accuracy", "94.5%"),
                mo.stat("Time", "5m 23s"),
            ]
        )

    # Layout
    return mo.vstack(
        [
            mo.md("## 🏋️ Model Training"),
            mo.hstack(
                [
                    # Left: Configuration
                    mo.vstack(
                        [
                            mo.md("### Configuration"),
                            model_selector,
                            epochs,
                            batch_size,
                            learning_rate,
                            mo.hstack([start_btn]),
                            mo.hstack([stop_btn]),
                        ],
                        align="stretch",
                    ),
                    # Right: Progress and Results
                    mo.vstack(
                        [mo.md("### Progress"), progress, status, results],
                        align="stretch",
                    ),
                ],
                widths=[1, 2],
            ),
        ]
    )
