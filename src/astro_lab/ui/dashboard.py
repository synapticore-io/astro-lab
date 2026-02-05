"""
AstroLab Dashboard - Redirect zur neuen App
"""


def create_astrolab_dashboard(*args, **kwargs):
    """Redirect zur neuen app.py Struktur"""
    print("ℹ️  AstroLab Dashboard wurde zur neuen app.py Struktur migriert.")
    print("   Verwende: marimo run src/astro_lab/ui/app.py")

    from astro_lab.ui.app import app

    return app
