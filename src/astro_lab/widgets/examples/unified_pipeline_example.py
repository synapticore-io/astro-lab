"""Example demonstrating the unified visualization pipeline.

This example shows how different backends are automatically orchestrated
to create optimal visualizations for astronomical data.
"""

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord

from astro_lab.tensors import SpatialTensorDict
from astro_lab.widgets import (
    create_publication_figure,
    create_visualization,
    plot_cosmic_web,
    plot_stellar_data,
    unified_pipeline,
)


def example_cosmic_web_visualization():
    """Example: Multi-backend cosmic web visualization with effects."""
    print("=== Cosmic Web Visualization Example ===")

    # Generate sample cosmic web data
    n_nodes = 10000
    n_clusters = 5

    # Create cluster centers in Mpc
    cluster_centers = np.random.randn(n_clusters, 3) * 100 * u.Mpc

    # Generate points around clusters
    points = []
    cluster_labels = []

    for i, center in enumerate(cluster_centers):
        # Points per cluster
        n_points = n_nodes // n_clusters

        # Generate clustered points with astropy units
        cluster_points = np.random.randn(n_points, 3) * 10 * u.Mpc + center
        points.append(cluster_points)
        cluster_labels.extend([i] * n_points)

    # Combine all points
    positions = np.vstack(points)

    # Create edges for cosmic web structure
    from sklearn.neighbors import kneighbors_graph

    positions_value = positions.to(u.Mpc).value
    A = kneighbors_graph(positions_value, n_neighbors=6, mode="connectivity")
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

    # Create SpatialTensorDict with proper units
    spatial_data = SpatialTensorDict(
        {
            "coordinates": torch.tensor(positions_value, dtype=torch.float32),
            "edge_index": edge_index,
            "cluster_labels": torch.tensor(cluster_labels, dtype=torch.long),
        },
        coordinate_system="cartesian",
        unit="Mpc",
    )

    # Visualize with different effects combinations
    print("\n1. Scientific analysis (PyVista)")
    create_visualization(
        spatial_data,
        backend="pyvista",
        visualization_type="cosmic_web",
        show_axes=True,
        background_color="white",
    )

    print("\n2. Photorealistic rendering (Blender + effects)")
    plot_cosmic_web(
        spatial_data,
        effects=["glow", "volumetric", "bloom"],
        photorealistic=True,
        interactive=False,
        render_path="cosmic_web_render.png",
        resolution=(3840, 2160),  # 4K
        samples=256,
    )

    print("\n3. Interactive exploration (Open3D)")
    create_visualization(
        spatial_data,
        backend="open3d",
        visualization_type="cosmic_web",
        show_bounds=True,
        compute_normals=True,
    )

    print("\n4. Web-ready visualization (Plotly)")
    create_visualization(
        spatial_data,
        backend="plotly",
        visualization_type="cosmic_web",
        export_formats=["html"],
        output_name="cosmic_web_interactive",
    )

    print("\n5. Unified pipeline (all backends)")
    result_unified = unified_pipeline.create_visualization(
        spatial_data,
        visualization_type="cosmic_web",
        effects=["glow", "filaments"],
        export_formats=["png", "html", "ply"],
        output_dir="./cosmic_web_outputs",
    )

    return result_unified


def example_stellar_neighborhood():
    """Example: Stellar neighborhood with proper motion visualization."""
    print("\n=== Stellar Neighborhood Example ===")

    # Create realistic stellar data using astropy
    n_stars = 5000

    # Generate positions in parsecs around the Sun
    distances = np.random.gamma(2, 50, n_stars) * u.pc  # Distance distribution
    ra = np.random.uniform(0, 360, n_stars) * u.deg
    dec = np.random.uniform(-90, 90, n_stars) * u.deg

    # Create SkyCoord objects
    coords = SkyCoord(ra=ra, dec=dec, distance=distances, frame="icrs")

    # Convert to cartesian for visualization
    cartesian = coords.cartesian
    positions = np.vstack([cartesian.x, cartesian.y, cartesian.z]).T

    # Generate stellar properties
    temperatures = np.random.normal(5778, 1500, n_stars) * u.K  # Stellar temperatures
    luminosities = np.random.lognormal(0, 2, n_stars) * u.Lsun  # Luminosities

    # Proper motions
    pmra = np.random.normal(0, 20, n_stars) * u.mas / u.yr
    pmdec = np.random.normal(0, 20, n_stars) * u.mas / u.yr

    # Create tensor dict
    stellar_data = SpatialTensorDict(
        {
            "coordinates": torch.tensor(positions.value, dtype=torch.float32),
            "temperature": torch.tensor(temperatures.value, dtype=torch.float32),
            "luminosity": torch.tensor(luminosities.value, dtype=torch.float32),
            "pmra": torch.tensor(pmra.value, dtype=torch.float32),
            "pmdec": torch.tensor(pmdec.value, dtype=torch.float32),
        },
        coordinate_system="icrs",
        unit="pc",
    )

    # Visualize with color and size mapping
    result = plot_stellar_data(
        stellar_data,
        color_by="temperature",  # Color by temperature (blackbody)
        size_by="luminosity",  # Size by luminosity
        show_vectors=True,  # Show proper motion vectors
        vector_property="pm",  # Proper motion vectors
        effects=["glow"],  # Stellar glow effect
        export_formats=["png", "html"],
        output_name="stellar_neighborhood",
    )

    return result


def example_galaxy_morphology():
    """Example: Galaxy visualization with morphological features."""
    print("\n=== Galaxy Morphology Example ===")

    # Generate spiral galaxy structure
    n_points = 50000
    n_arms = 4

    # Spiral parameters
    angles = np.random.uniform(0, 4 * np.pi, n_points)
    radii = np.random.exponential(scale=10, size=n_points) * u.kpc

    # Add spiral structure
    arm_angles = angles + (2 * np.pi / n_arms) * np.floor(
        n_arms * np.random.uniform(0, 1, n_points)
    )

    # Convert to cartesian with some vertical spread
    x = radii * np.cos(arm_angles)
    y = radii * np.sin(arm_angles)
    z = np.random.normal(0, 0.5, n_points) * u.kpc  # Disk thickness

    positions = np.vstack([x.value, y.value, z.value]).T

    # Add bulge component
    n_bulge = 10000
    bulge_positions = np.random.randn(n_bulge, 3) * 2  # kpc, concentrated center

    # Combine disk and bulge
    all_positions = np.vstack([positions, bulge_positions])

    # Star formation regions (higher in spiral arms)
    sfr = np.exp(-radii.value / 5) * np.abs(np.sin(n_arms * angles))
    sfr_bulge = np.zeros(n_bulge)
    all_sfr = np.concatenate([sfr, sfr_bulge])

    # Create galaxy data
    galaxy_data = SpatialTensorDict(
        {
            "coordinates": torch.tensor(all_positions, dtype=torch.float32),
            "star_formation_rate": torch.tensor(all_sfr, dtype=torch.float32),
            "component": torch.cat(
                [
                    torch.zeros(n_points, dtype=torch.long),  # Disk
                    torch.ones(n_bulge, dtype=torch.long),  # Bulge
                ]
            ),
        },
        coordinate_system="galactocentric",
        unit="kpc",
    )

    # Create multi-view visualization
    result = unified_pipeline.create_visualization(
        galaxy_data,
        visualization_type="galaxy",
        effects=["volumetric", "bloom"],
        color_by="star_formation_rate",
        colormap="magma",
        camera_views=["face-on", "edge-on", "perspective"],
        export_formats=["png", "blend"],
        output_name="spiral_galaxy",
    )

    return result


def example_publication_figure():
    """Example: Create publication-ready figure."""
    print("\n=== Publication Figure Example ===")

    # Generate sample data
    n_objects = 1000
    positions = np.random.randn(n_objects, 3) * 50  # Mpc
    redshifts = np.random.uniform(0, 2, n_objects)
    masses = np.random.lognormal(12, 1, n_objects)  # log10(M/M_sun)

    data = SpatialTensorDict(
        {
            "coordinates": torch.tensor(positions, dtype=torch.float32),
            "redshift": torch.tensor(redshifts, dtype=torch.float32),
            "mass": torch.tensor(masses, dtype=torch.float32),
        },
        coordinate_system="comoving",
        unit="Mpc",
    )

    # Create different figure styles
    styles = ["scientific", "presentation", "poster"]

    for style in styles:
        output_path = f"figure_{style}.png"

        create_publication_figure(
            data,
            output_path=output_path,
            figure_type="panel",
            style=style,
            visualization_type="cosmic_web",
            color_by="redshift",
            size_by="mass",
            show_colorbar=True,
            show_scalebar=True,
            title=f"Large Scale Structure ({style.capitalize()} Style)",
        )

        print(f"Created {style} figure: {output_path}")


def example_astropy_integration():
    """Example: Deep integration with astropy units and coordinates."""
    print("\n=== Astropy Integration Example ===")

    # Use astropy cosmology for proper distance calculations
    from astropy.cosmology import Planck18 as cosmo

    # Generate objects at different redshifts
    redshifts = np.logspace(-2, 1, 100)

    # Calculate comoving distances
    comoving_distances = cosmo.comoving_distance(redshifts)

    # Random sky positions
    ra = np.random.uniform(0, 360, 100) * u.deg
    dec = np.random.uniform(-60, 60, 100) * u.deg

    # Convert to 3D comoving coordinates
    coords = SkyCoord(ra=ra, dec=dec, distance=comoving_distances, frame="icrs")

    # Get cartesian coordinates
    cartesian = coords.cartesian
    positions = np.vstack(
        [cartesian.x.to(u.Mpc), cartesian.y.to(u.Mpc), cartesian.z.to(u.Mpc)]
    ).T

    # Calculate physical properties
    luminosity_distances = cosmo.luminosity_distance(redshifts)
    angular_diameter_distances = cosmo.angular_diameter_distance(redshifts)

    # Create data with full astropy integration
    cosmo_data = {
        "coordinates": positions,
        "redshift": redshifts,
        "luminosity_distance": luminosity_distances,
        "angular_diameter_distance": angular_diameter_distances,
        "ra": ra,
        "dec": dec,
    }

    # Visualize with automatic unit handling
    result = create_visualization(
        cosmo_data,
        visualization_type="cosmic_web",
        color_by="redshift",
        colormap="spectral",
        show_axes=True,
        axis_labels=["X [Mpc]", "Y [Mpc]", "Z [Mpc]"],
        title="Cosmological Distance Ladder",
    )

    return result


def main():
    """Run all examples."""
    print("AstroLab Unified Visualization Pipeline Examples")
    print("=" * 50)

    # Run examples
    example_cosmic_web_visualization()
    example_stellar_neighborhood()
    example_galaxy_morphology()
    example_publication_figure()
    example_astropy_integration()

    print("\n✨ All examples completed!")


if __name__ == "__main__":
    main()
