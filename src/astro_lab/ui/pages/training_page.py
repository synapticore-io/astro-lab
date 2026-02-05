"""
Training Page - Using existing AstroLab training components
Following Marimo 2025 best practices with existing components
"""

import marimo as mo

from ..components import state

# Use existing AstroLab components with relative imports
from ..components.trainer import create_training_form, create_training_status


def create_training_page():
    """Create training page using existing AstroLab components"""

    # Check if data is loaded in global state
    current_state = state.get_state()
    loaded_data = current_state.get("loaded_data")

    if loaded_data is None:
        return mo.vstack(
            [
                mo.md("# 🚀 Model Training"),
                mo.md("⚠️ **No data loaded!**"),
                mo.md("Go to the **Data** tab and load survey data first."),
                mo.md(
                    "Training requires processed astronomical data with features and coordinates."
                ),
            ]
        )

    # Use existing training components
    training_form = create_training_form()
    training_status = create_training_status()

    # Additional visualization for training results
    training_results_viz = create_training_results_visualization()

    return mo.vstack(
        [
            mo.md("# 🚀 Model Training"),
            mo.md(
                "Train astronomical machine learning models using AstroLab's training framework."
            ),
            mo.md("---"),
            # Data Status
            mo.md("## 📊 Training Data Status"),
            create_data_status_display(loaded_data),
            mo.md("---"),
            # Training Configuration
            mo.md("## ⚙️ Training Configuration"),
            training_form,
            mo.md("---"),
            # Training Status
            mo.md("## 📈 Training Status"),
            training_status,
            mo.md("---"),
            # Results Visualization
            mo.md("## 🎨 Training Results"),
            training_results_viz,
        ]
    )


def create_data_status_display(loaded_data):
    """Display status of loaded training data"""

    if loaded_data is None or loaded_data.is_empty():
        return mo.md("❌ **No data available for training**")

    # Get data info
    try:
        survey = getattr(loaded_data, "survey", "Unknown")
        num_features = getattr(loaded_data, "num_features", "Unknown")
        num_classes = getattr(loaded_data, "num_classes", "Unknown")

        status_md = """
        ✅ **Training data ready!**
        
        - **Survey:** {}
        - **Features:** {}
        - **Classes:** {}
        - **Data Module:** {}
        """.format(survey, num_features, num_classes, type(loaded_data).__name__)

        return mo.md(status_md)

    except Exception as e:
        return mo.md("⚠️ **Data loaded but could not read details:** " + str(e))


def create_training_results_visualization():
    """Create visualization of training results"""

    current_state = state.get_state()
    training_result = current_state.get("training_result")
    trained_model = current_state.get("trained_model")

    if training_result is None:
        return mo.md("⏳ **No training completed yet**")

    if trained_model is None:
        return mo.md("⚠️ **Training completed but no model available**")

    # Show training completion info
    model_type = type(trained_model).__name__
    trainer_type = type(training_result).__name__

    results_md = """
    ## ✅ Training Completed Successfully!
    
    **Model:** {}
    **Trainer:** {}
    **Status:** {}
    
    💡 Model is ready for inference and evaluation.
    """.format(model_type, trainer_type, current_state.get("status", "Completed"))

    # Add model visualization button
    viz_button = mo.ui.button("🎨 Visualize Model Results", kind="success")

    viz_content = mo.md("⏳ **Click 'Visualize Model Results' to generate plots**")

    if viz_button.value:
        viz_content = create_model_visualization(trained_model, training_result)

    return mo.vstack([mo.md(results_md), viz_button, viz_content])


def create_model_visualization(model, trainer):
    """Create visualization for trained model"""

    try:
        # Get model info
        model_name = type(model).__name__

        # Create simple model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info = """
        ### 🧠 Model Architecture Summary
        
        **Model Type:** {}
        **Total Parameters:** {:,}
        **Trainable Parameters:** {:,}
        **Parameter Efficiency:** {:.2%}
        """.format(
            model_name,
            total_params,
            trainable_params,
            trainable_params / total_params if total_params > 0 else 0,
        )

        # Try to get some training metrics if available
        metrics_info = "📊 **Training metrics visualization would appear here**"

        # Check if trainer has logged metrics
        if hasattr(trainer, "logged_metrics"):
            metrics = trainer.logged_metrics
            if metrics:
                metrics_md = "### 📈 Training Metrics\n\n"
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_md += "- **{}:** {:.4f}\n".format(key, value)
                metrics_info = metrics_md

        return mo.vstack(
            [
                mo.md(model_info),
                mo.md(metrics_info),
                mo.md("🎯 **Model ready for prediction and evaluation tasks**"),
            ]
        )

    except Exception as e:
        return mo.md("⚠️ **Could not visualize model:** " + str(e))
