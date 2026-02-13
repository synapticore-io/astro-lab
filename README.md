# 🌌 AstroLab - Astro GNN Laboratory for Cosmic Web Exploration

A comprehensive **Astro GNN laboratory** for exploring cosmic web structures through graph neural networks, astronomical data analysis, and interactive 3D visualization across multiple astronomical scales.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/bjoernbethge/astro-lab.git
cd astro-lab
uv sync
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### First Steps
```bash
# Download data (optional - data may already be available)
astro-lab download gaia

# Preprocess data (recommended first step)
astro-lab preprocess --survey gaia --max-samples 1000

# Analyze cosmic web structure
astro-lab cosmic-web gaia --max-samples 10000 --clustering-scales 5 10 25 --visualize

# Start interactive development environment
marimo run src/astro_lab/ui/app.py
```

## 📊 AstroLab Consolidated Catalog

AstroLab provides a **consolidated multi-survey catalog** with cosmic web structure classifications, combining data from Gaia, SDSS, and 2MASS.

### Generate the Catalog

```bash
# Generate with default settings (Gaia only, 10k samples)
python scripts/generate_astrolab_catalog.py --max-samples 10000

# Generate with multiple surveys
python scripts/generate_astrolab_catalog.py --surveys gaia sdss twomass --max-samples 50000

# Generate with custom clustering scales
python scripts/generate_astrolab_catalog.py --clustering-scales 5 10 25 50 100
```

### Features Included

- **Multi-wavelength photometry**: Optical (Gaia, SDSS) + Near-IR (2MASS)
- **Astrometry**: Positions, proper motions, parallaxes
- **3D Coordinates**: Cartesian coordinates in parsecs
- **Cosmic Web Classification**: Filament, void, node, field at multiple scales
- **Density Fields**: Local density estimates at each scale
- **Structure Metrics**: Anisotropy, connectivity, topology

### Using the Catalog

```python
import polars as pl

# Load the catalog
catalog = pl.read_parquet("data/catalogs/astrolab_catalog_v1.parquet")

# Filter by structure type (e.g., filaments at 10 pc scale)
filaments = catalog.filter(pl.col("cosmic_web_class_10.0pc") == 1)

# Get high-density regions
high_density = catalog.filter(pl.col("density_10.0pc") > threshold)
```

### Generate Visualizations

```bash
# Create interactive 3D plots and statistical visualizations
python scripts/generate_visualizations.py

# Output: data/visualizations/*.html
```

See the [complete catalog documentation](data/README.md) and [example usage](examples/use_astrolab_catalog.py) for more details.


## 🎨 Visual Examples

AstroLab provides powerful interactive 3D visualizations of cosmic web structures across multiple astronomical scales.

### Quick Visualization

Generate publication-quality visualizations from the AstroLab catalog:

```bash
# Generate the catalog first
python scripts/generate_astrolab_catalog.py --max-samples 10000

# Create visualizations
python scripts/generate_visualizations.py

# Open the generated HTML files in data/visualizations/
```

### Available Visualizations

- **Interactive 3D Cosmic Web**: Real-time exploration using Plotly with structure classification coloring
- **Multi-Scale Comparison**: Side-by-side views of structures at different clustering scales
- **Structure Distribution**: Statistical analysis of filaments, voids, nodes, and field regions
- **Density Maps**: Local density field visualizations

### Programmatic Usage

```python
from astro_lab.data.analysis.cosmic_web import ScalableCosmicWebAnalyzer
import polars as pl

# Analyze cosmic web structure
analyzer = ScalableCosmicWebAnalyzer()
catalog = pl.read_parquet("data/catalogs/astrolab_catalog_v1.parquet")

# Extract coordinates
coordinates = catalog.select(['x', 'y', 'z']).to_numpy()

# Run analysis
results = analyzer.analyze_cosmic_web(
    coordinates,
    scales=[5.0, 10.0, 25.0],
    use_adaptive_sampling=True
)
```

Or use the interactive UI:
```bash
marimo run src/astro_lab/ui/app.py
```

**Live Demo**: Visit our [GitHub Pages documentation](https://bjoernbethge.github.io/astro-lab/) for API documentation and guides (Documentation auto-deployed on every commit to main).

## 🧠 Astro GNN Models & Tasks

### **Core GNN Models**
AstroLab provides ready-to-use model factory functions for all main tasks:

- **AstroGraphGNN**: Spatial graph neural networks for cosmic web structure detection
  - Factory: `create_astro_graph_gnn(num_features, num_classes, **kwargs)`
  - Example:
    ```python
    from astro_lab.models.astro_model import create_astro_graph_gnn
    model = create_astro_graph_gnn(num_features=16, num_classes=3)
    ```
- **AstroNodeGNN**: Node classification for stellar/galaxy properties
  - Factory: `create_astro_node_gnn(num_features, num_classes, **kwargs)`
  - Example:
    ```python
    from astro_lab.models.astro_model import create_astro_node_gnn
    model = create_astro_node_gnn(num_features=8, num_classes=5)
    ```
- **AstroPointNet**: Point cloud processing for 3D astronomical data
  - Factory: `create_astro_pointnet(num_features, num_classes, **kwargs)`
  - Example:
    ```python
    from astro_lab.models.astro_model import create_astro_pointnet
    model = create_astro_pointnet(num_features=3, num_classes=2)
    ```
- **AstroTemporalGNN**: Time-series analysis for variable objects
  - Factory: `create_astro_temporal_gnn(num_features, num_classes, **kwargs)`
  - Example:
    ```python
    from astro_lab.models.astro_model import create_astro_temporal_gnn
    model = create_astro_temporal_gnn(num_features=12, num_classes=4)
    ```

Legacy/specialized factories (for compatibility):
- `create_cosmic_web_model` → use `create_astro_graph_gnn`
- `create_stellar_model` → use `create_astro_node_gnn`
- `create_galaxy_model`, `create_exoplanet_model` for specialized galaxy/exoplanet tasks

### **Primary Tasks**
- **Cosmic Web Clustering**: Multi-scale structure detection (stellar to galactic scales)
- **Filament Detection**: MST, Morse theory, and Hessian eigenvalue analysis
- **Stellar Classification**: Spectral type and evolutionary stage prediction
- **Galaxy Morphology**: Shape and structure classification
- **Exoplanet Host Analysis**: Stellar neighborhood clustering
- **Temporal Variability**: Light curve analysis and period detection

### **Multi-Scale Analysis**
- **Stellar Scale** (1-100 parsecs): Local galactic disk structure
- **Galactic Scale** (1-100 Megaparsecs): Galaxy clusters and superclusters  
- **Exoplanet Scale** (10-500 parsecs): Stellar neighborhoods and associations

## 🌟 Key Features

### 🔬 **Multi-Survey Data Integration**
- **Gaia DR3**: Stellar catalogs with proper motions and cosmic web clustering
- **SDSS**: Galaxy surveys and spectra with large-scale structure analysis
- **NSA**: Galaxy catalogs with distances and cosmic web visualization  
- **TNG50**: Cosmological simulations with filament detection
- **NASA Exoplanet Archive**: Confirmed exoplanets with host star clustering
- **LINEAR**: Asteroid light curves with orbital family analysis

### 🌌 **Cosmic Web Analysis**

#### **Interactive 3D Visualization**
- **CosmographBridge**: Real-time cosmic web visualization with physics simulation
- **Survey-specific colors**: Gold for stars, blue for galaxies, green for simulations
- **Multi-backend support**: PyVista, Open3D, Blender, and Plotly integration
- **Live tensor sync**: Real-time updates between analysis and visualization

#### **Specialized Tensor Operations**
```python
from astro_lab.tensors import SpatialTensorDict

# Create spatial tensor with coordinate system support
spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="parsec")

# Multi-scale cosmic web clustering  
labels = spatial.cosmic_web_clustering(eps_pc=10.0, min_samples=5)

# Grid-based structure analysis
structure = spatial.cosmic_web_structure(grid_size_pc=100.0)

# Local density computation
density = spatial.analyze_local_density(radius_pc=50.0)
```

## 📚 Documentation & API Reference

The complete, up-to-date documentation is available as a modern website:

- **[API Reference](./docs/api/astro_lab/)**
- **[Cosmic Web Guide](./docs/cosmic_web_guide.md)**
- **[Model Documentation](./docs/api/astro_lab.models.md)**
- **[Training Guide](./docs/api/astro_lab.training.md)**

All code is fully documented with mkdocstrings and includes automatic class inheritance diagrams, usage examples, and configuration options.

> **ℹ️ Automatic Documentation Deployment**
>
> The documentation is automatically generated and deployed to GitHub Pages on every push to the `main` branch using a [GitHub Action](.github/workflows/docs.yml). You do not need to build or deploy the documentation manually—simply push your changes to `main` and the latest docs will be published automatically.

## 🛠️ CLI Reference

AstroLab provides a comprehensive command-line interface for all aspects of astronomical machine learning and cosmic web analysis.

### Core Commands

```bash
# Show all available commands
astro-lab --help

# Get help for specific commands
astro-lab <command> --help
```

### Data Processing
```bash
# Download raw survey data
astro-lab download gaia
astro-lab download gaia --force
astro-lab download --list

# Preprocess a single survey
astro-lab preprocess gaia --max-samples 10000

# Preprocess with quality filtering
astro-lab preprocess gaia --force

# Preprocess with custom output directory
astro-lab preprocess gaia --output-dir ./processed_data
```

### Configuration Management
```bash
# Create new configuration file
astro-lab config create -o my_experiment.yaml --template gaia

# Show available survey configurations
astro-lab config surveys

# Show specific survey configuration details
astro-lab config show gaia
astro-lab config show nsa
```

### Model Training
```bash
# Train with configuration file
astro-lab train -c my_experiment.yaml --verbose

# Train with command-line parameters
astro-lab train --survey gaia --model gcn --epochs 50 --batch-size 32
astro-lab train --survey nsa --model astro_node_gnn --learning-rate 0.001 --devices 2

# Resume from checkpoint
astro-lab train -c config.yaml --checkpoint path/to/checkpoint.ckpt

# Debug training with small dataset
astro-lab train --survey gaia --max-samples 1000 --overfit-batches 10
```

### Hyperparameter Optimization
```bash
# Optimize hyperparameters
astro-lab optimize gaia --trials 50 --timeout 3600
astro-lab optimize gaia --trials 100

# Quick optimization for debugging
astro-lab optimize gaia --trials 10 --max-samples 1000
```

### Cosmic Web Analysis
```bash
# Multi-scale stellar structure analysis
astro-lab cosmic-web gaia --max-samples 100000 --clustering-scales 5 10 25 50 --visualize

# Large-scale galaxy structure  
astro-lab cosmic-web nsa --clustering-scales 5 10 20 50 --redshift-limit 0.15

# Exoplanet host star clustering
astro-lab cosmic-web exoplanet --clustering-scales 10 25 50 100 200 --min-samples 3

# Custom analysis with output directory
astro-lab cosmic-web gaia --catalog-path ./my_catalog.fits --output-dir ./results --verbose
```

### Supported Surveys
All commands support these astronomical surveys:
- `gaia`: [Gaia DR3](https://www.cosmos.esa.int/web/gaia/dr3) stellar catalog
- `sdss`: [Sloan Digital Sky Survey](https://www.sdss.org/)  
- `nsa`: [NASA-Sloan Atlas](http://nsatlas.org/) galaxy catalog
- `tng50`: [TNG50](https://www.tng-project.org/) cosmological simulation
- `exoplanet`: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- `rrlyrae`: [RR Lyrae](https://www.sdss.org/dr17/spectro/vac/) variable stars
- `linear`: [LINEAR](https://en.wikipedia.org/wiki/Lincoln_Near-Earth_Asteroid_Research) asteroid survey

## 🔧 Setup Scripts

AstroLab provides automated setup scripts for easy installation across different platforms.

### Linux/macOS Setup (setup.sh)

The `setup.sh` script automates the entire installation process on Linux and macOS systems:

```bash
# Make the script executable and run it
chmod +x setup.sh
./setup.sh
```

**What the script does:**
1. **Installs uv package manager** if not already present
2. **Runs `uv sync`** to install all dependencies from `pyproject.toml`
3. **Installs PyTorch Geometric extensions** for CUDA support
4. **Activates the virtual environment** automatically
5. **Provides instructions** for future activation

**Manual equivalent:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Install dependencies
uv sync
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Activate environment
source .venv/bin/activate
```

### Windows Setup (setup.ps1)

The `setup.ps1` script provides the same functionality for Windows PowerShell:

```powershell
# Run the PowerShell setup script
.\setup.ps1
```

**What the script does:**
1. **Installs uv package manager** via PowerShell
2. **Runs `uv sync`** to install dependencies
3. **Installs PyTorch Geometric extensions** with CUDA support
4. **Activates the virtual environment**
5. **Provides activation instructions** for future use

**Manual equivalent (PowerShell):**
```powershell
# Install uv
irm https://astral.sh/uv/install.ps1 | iex
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# Install dependencies
uv sync
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Activate environment
.\.venv\Scripts\Activate.ps1
```

### Environment Activation

After setup, activate the environment for future sessions:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

### Verification

Test your installation:
```bash
# Check CLI availability
astro-lab --help

# Verify CUDA support (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test cosmic web analysis
astro-lab cosmic-web gaia --max-samples 100 --clustering-scales 5 10
```

## 📖 Documentation Generation

AstroLab includes automated documentation generation and management tools.

### Documentation Scripts

The `docs/generate_docs.py` script provides comprehensive documentation management:

```bash
# Generate/update all documentation
python docs/generate_docs.py update

# Start local documentation server
python docs/generate_docs.py serve

# Deploy documentation to GitHub Pages
python docs/generate_docs.py deploy
```

**What the documentation script does:**
1. **Scans source code** for all Python modules
2. **Generates API documentation** using mkdocstrings
3. **Creates navigation structure** automatically
4. **Builds documentation** with MkDocs
5. **Serves locally** for development
6. **Deploys to GitHub Pages** for production

### Manual Documentation Commands

You can also run documentation commands directly:

```bash
# Install documentation dependencies
uv run pip install mkdocs mkdocstrings[python] mkdocs-material

# Build documentation
uv run mkdocs build --clean

# Serve documentation locally (http://127.0.0.1:8000)
uv run mkdocs serve

# Deploy to GitHub Pages
uv run mkdocs gh-deploy --force
```

### Documentation Structure

The documentation system automatically generates:
- **API Reference**: Complete code documentation with inheritance diagrams
- **Cosmic Web Guide**: Comprehensive analysis tutorials
- **Model Documentation**: GNN architecture and training guides
- **Configuration Reference**: All survey and model configurations

## 🤖 Automation and UI Scripts

AstroLab automation is handled through:

1. **Setup Scripts**: `setup.sh` and `setup.ps1` for environment setup
2. **Documentation Scripts**: `docs/generate_docs.py` for documentation management  
3. **UI Launch Script**: `run_ui.py` for starting the interactive dashboard
4. **CLI Commands**: Built-in automation through the `astro-lab` CLI

### UI Launch Script

The `run_ui.py` script provides an easy way to start the AstroLab interactive dashboard:

```bash
# Start the AstroLab UI dashboard
python run_ui.py

# The dashboard will be available at http://localhost:2718
```

**What the UI script does:**
- Launches the Marimo reactive notebook interface
- Provides access to cosmic web analysis tools
- Enables interactive data visualization  
- Runs on port 2718 by default

## 🏗️ Architecture

### Core Components
```
astro-lab/
├── src/astro_lab/
│   ├── cli/
│   │   ├── cosmic_web.py      # Cosmic web CLI interface
│   │   ├── train.py           # Model training CLI
│   │   └── optimize.py        # Hyperparameter optimization
│   ├── data/
│   │   ├── cosmic_web.py      # Core cosmic web analysis
│   │   └── datasets/          # Survey-specific datasets
│   ├── tensors/
│   │   └── tensordict_astro.py # Spatial tensor operations  
│   ├── models/
│   │   ├── core/              # GNN model implementations
│   │   └── components/        # Model building blocks
│   ├── widgets/
│   │   ├── cosmograph_bridge.py   # Interactive visualization
│   │   ├── graph.py               # Graph analysis functions
│   │   └── plotly_bridge.py       # 3D plotting
│   ├── ui/modules/
│   │   ├── cosmic_web.py      # UI for cosmic web analysis
│   │   ├── analysis.py        # Interactive analysis tools
│   │   └── visualization.py   # Visualization interface
│   └── training/              # Training framework
├── configs/                   # Configuration files
├── docs/
│   ├── cosmic_web_guide.md    # Comprehensive cosmic web guide
│   └── api/                   # Auto-generated API documentation
└── test/                      # Test suite
```

### Key Dependencies
- **PyTorch 2.7.1+cu128**: GPU-accelerated deep learning with geometric extensions
- **PyTorch Geometric**: Graph neural networks for cosmic web analysis
- **Lightning 2.5.1**: Training framework with MLflow integration
- **Polars 1.31.0**: High-performance data processing
- **AstroPy 7.1.0**: Astronomical calculations and coordinate systems
- **Cosmograph**: Interactive graph visualization with physics simulation
- **Marimo**: Reactive notebooks for interactive analysis
- **scikit-learn**: Clustering algorithms (DBSCAN, K-means, etc.)

## 🛠️ Framework Stack

AstroLab is built on cutting-edge frameworks and libraries for astronomical machine learning:

### **Core ML/DL Frameworks**
- **[PyTorch](https://pytorch.org/)** - GPU-accelerated deep learning with CUDA support
- **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** - Graph neural networks for spatial data
- **[PyTorch Lightning](https://lightning.ai/)** - Professional training framework with MLflow integration
- **[Optuna](https://optuna.org/)** - Hyperparameter optimization and experiment tracking

### **Data Processing & Visualization**
- **[Polars](https://pola.rs/)** - Lightning-fast data processing with Rust backend
- **[Plotly](https://plotly.com/python/)** - Interactive web-based visualizations
- **[PyVista](https://docs.pyvista.org/)** - 3D scientific visualization
- **[Open3D](http://www.open3d.org/)** - Real-time 3D point cloud processing

### **Astronomical Libraries**
- **[AstroPy](https://www.astropy.org/)** - Core astronomical calculations and coordinate systems
- **[AstromL](https://www.astroml.org/)** - Machine learning for astronomy
- **[AstroQuery](https://astroquery.readthedocs.io/)** - Astronomical data access
- **[SDSS Access](https://sdss-access.readthedocs.io/)** - Sloan Digital Sky Survey data

### **Interactive Development**
- **[Marimo](https://marimo.io/)** - Reactive Python notebooks for interactive analysis
- **[Cosmograph](https://cosmograph.app/)** - Interactive graph visualization with physics simulation
- **[Blender Python API](https://docs.blender.org/api/current/)** - Professional 3D rendering and animation

### **Development Tools**
- **[UV](https://docs.astral.sh/uv/)** - Fast Python package manager and resolver
- **[Ruff](https://docs.astral.sh/ruff/)** - Extremely fast Python linter
- **[MyPy](https://mypy-lang.org/)** - Static type checking for Python
- **[Pre-commit](https://pre-commit.com/)** - Git hooks for code quality

## 🎯 Use Cases

### Stellar Structure Analysis
```python
from astro_lab.data.cosmic_web import analyze_gaia_cosmic_web

# Analyze local stellar neighborhoods
results = analyze_gaia_cosmic_web(
    max_samples=100000,
    magnitude_limit=12.0,
    clustering_scales=[5.0, 10.0, 25.0, 50.0],  # parsecs
    min_samples=5
)

print(f"Found {results['n_stars']} stars")
for scale, stats in results['clustering_results'].items():
    print(f"{scale}: {stats['n_clusters']} clusters, {stats['grouped_fraction']:.1%} grouped")
```

### Galaxy Cluster Analysis
```python
from astro_lab.data.cosmic_web import analyze_nsa_cosmic_web

# Large-scale structure analysis
results = analyze_nsa_cosmic_web(
    redshift_limit=0.15,
    clustering_scales=[5.0, 10.0, 20.0, 50.0],  # Mpc
    min_samples=5
)
```

### Interactive Cosmic Web Visualization
```python
from astro_lab.widgets.cosmograph_bridge import CosmographBridge
from astro_lab.data.cosmic_web import CosmicWebAnalyzer

# Load and analyze data
analyzer = CosmicWebAnalyzer()
results = analyzer.analyze_gaia_cosmic_web(max_samples=10000)

# Create interactive 3D visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")

# Display with physics simulation and survey-specific colors
widget.show()  # Gold points for Gaia stars with real-time clustering
```

### Filament Detection
```python
from astro_lab.tensors import SpatialTensorDict
from astro_lab.data.cosmic_web import CosmicWebAnalyzer

# Create spatial tensor
spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="parsec")

# Detect filamentary structures
analyzer = CosmicWebAnalyzer()
filaments = analyzer.detect_filaments(
    spatial, 
    method="mst",  # or "morse_theory", "hessian"
    n_neighbors=20,
    distance_threshold=10.0
)

print(f"Detected {filaments['n_filament_segments']} filament segments")
print(f"Total filament length: {filaments['total_filament_length']:.1f} pc")
```

### Multi-Backend Visualization
```python
from astro_lab.widgets.tensor_bridge import create_tensor_bridge

# Create visualization bridge
bridge = create_tensor_bridge(backend="cosmograph")  # or "pyvista", "blender"

# Visualize cosmic web with clustering
viz = bridge.cosmic_web_to_backend(
    spatial_tensor=spatial,
    cluster_labels=labels,
    point_size=2.0,
    show_filaments=True
)
```

## 🔧 Development

### Interactive Development
```bash
# Start Marimo reactive notebook with cosmic web UI
uv run marimo run src/astro_lab/ui/app.py

# Launch MLflow UI for experiment tracking
uv run mlflow ui --backend-store-uri ./data/experiments
```

### Testing
```bash
# Run full test suite
uv run pytest -v

# Test specific components
uv run pytest test/test_cosmic_web.py -v
uv run pytest test/test_data.py -v
uv run pytest test/test_lightning.py -v
uv run pytest src/astro_lab/tensors/ -v -k cosmic_web
```

## 📊 Experiment Tracking

All cosmic web analyses are automatically tracked with MLflow:

```python
# Results are logged with cosmic web metadata
- clustering_scales: [5.0, 10.0, 25.0, 50.0]
- survey_type: "gaia" 
- n_clusters_per_scale: {5.0: 125, 10.0: 89, 25.0: 45, 50.0: 12}
- filament_detection_method: "mst"
- visualization_backend: "cosmograph"
```

## 🎨 Visualization Gallery

### Supported Backends
- **Cosmograph**: Interactive 3D with physics simulation and survey-specific colors
- **PyVista**: High-quality 3D rendering with filament visualization  
- **Plotly**: Web-based interactive plots with multi-scale clustering
- **Blender**: Professional 3D rendering and animation via albpy integration
- **Open3D**: Real-time point cloud visualization with octree support

### Example Visualizations
- **Gaia stellar neighborhoods**: Gold points with gravitational clustering
- **NSA galaxy superclusters**: Blue points with large-scale structure
- **TNG50 cosmic web**: Green points with dark matter filaments
- **Exoplanet host clusters**: Magenta points with stellar associations

## 🤝 Contributing

We welcome contributions to cosmic web analysis features! See our [contribution guidelines](CONTRIBUTING.md) for details on:

- Adding new filament detection algorithms
- Implementing additional clustering methods  
- Creating visualization backends
- Extending survey support
- Improving performance with GPU acceleration

## 🙏 Acknowledgments

Special thanks to the **Astro Graph Agent Team** for their invaluable contributions to the advanced visualization modules and cosmic web analysis features. Their expertise in astronomical data visualization and graph neural networks has been instrumental in making AstroLab a comprehensive Astro GNN laboratory.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**AstroLab** - An Astro GNN laboratory for exploring cosmic web structures across all scales of the universe. 🌌✨ 

## 🛠️ CLI Command Overview

| Command         | Beschreibung / Zweck                                 |
|----------------|------------------------------------------------------|
| download       | Lade Rohdaten eines Surveys herunter                 |
| preprocess     | Verarbeite Rohdaten zu ML-tauglichen Formaten        |
| train          | Trainiere ein Modell auf Survey-Daten                |
| optimize       | Hyperparameter-Optimierung für ein Modell            |
| info           | Zeige Metadaten, Spalten, Beispiele, Validierung     |
| cosmic-web     | Analysiere kosmische Netzwerke/Strukturen            |
| config         | Konfigurationsdateien anzeigen/erstellen/validieren  |
| build-dataset  | Erzeuge ML-Ready Dataset aus harmonisierten Daten    |

### Data Inspection & Info
```bash
# Zeige Übersicht aller verfügbaren Surveys
astro-lab info

# Zeige Metadaten für einen Survey
astro-lab info gaia

# Zeige Spalteninformationen
astro-lab info gaia --columns

# Zeige Beispielzeilen
astro-lab info gaia --sample 5

# Führe Datenvalidierung durch
astro-lab info gaia --validate
``` 