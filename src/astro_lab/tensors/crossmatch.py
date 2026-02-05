"""
Cross-matching TensorDict for astronomical catalogs.

TensorDict for cross-matching between astronomical catalogs with proper
distance calculations and match quality assessment.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from .base import AstroTensorDict
from .mixins import ValidationMixin


class CrossMatchTensorDict(AstroTensorDict, ValidationMixin):
    """
    TensorDict for cross-matching between astronomical catalogs.

    Structure:
    {
        "catalog1": AstroTensorDict,  # First catalog
        "catalog2": AstroTensorDict,  # Second catalog
        "matches": Tensor[M, 2],      # Match indices [idx1, idx2]
        "distances": Tensor[M],       # Angular distances
        "match_quality": Tensor[M],   # Match quality
        "meta": {
            "match_radius": float,
            "algorithm": str,
            "n_matches": int,
            "match_statistics": Dict,
        }
    }
    """

    def __init__(
        self,
        catalog1: AstroTensorDict,
        catalog2: AstroTensorDict,
        match_radius: float = 1.0,
        algorithm: str = "nearest_neighbor",
        **kwargs,
    ):
        """
        Initialize CrossMatchTensorDict.

        Args:
            catalog1: First catalog data
            catalog2: Second catalog data
            match_radius: Search radius in arcseconds
            algorithm: Matching algorithm
        """
        # Validate that both catalogs contain spatial information
        if not self._has_spatial_info(catalog1):
            raise ValueError("Catalog1 must contain spatial information")
        if not self._has_spatial_info(catalog2):
            raise ValueError("Catalog2 must contain spatial information")

        # Initialize empty matches
        empty_matches = torch.empty((0, 2), dtype=torch.long)
        empty_distances = torch.empty(0)
        empty_quality = torch.empty(0)

        data = {
            "catalog1": catalog1,
            "catalog2": catalog2,
            "matches": empty_matches,
            "distances": empty_distances,
            "match_quality": empty_quality,
            "meta": {
                "match_radius": match_radius,
                "algorithm": algorithm,
                "n_matches": 0,
                "match_statistics": {},
            },
        }

        super().__init__(data, **kwargs)

    def _has_spatial_info(self, catalog: AstroTensorDict) -> bool:
        """Check if catalog contains spatial information."""
        return (
            "coordinates" in catalog
            or "ra" in catalog
            and "dec" in catalog
            or hasattr(catalog, "coordinates")
        )

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract cross-match features from the TensorDict.

        Args:
            feature_types: Types of features to extract ('crossmatch', 'statistics', 'quality')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted cross-match features
        """
        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Add cross-match specific features
        if feature_types is None or "crossmatch" in feature_types:
            # Basic cross-match properties
            features["n_matches"] = torch.tensor(
                self["meta"]["n_matches"], dtype=torch.float32
            )
            features["match_radius"] = torch.tensor(
                self["meta"]["match_radius"], dtype=torch.float32
            )
            features["completeness"] = torch.tensor(
                self["meta"]["match_statistics"].get("completeness", 0.0),
                dtype=torch.float32,
            )

        if feature_types is None or "statistics" in feature_types:
            # Statistical features of matches
            if self["meta"]["n_matches"] > 0:
                distances = self["distances"]
                features["mean_distance"] = torch.mean(distances)
                features["std_distance"] = torch.std(distances)
                features["median_distance"] = torch.median(distances)
                features["min_distance"] = torch.min(distances)
                features["max_distance"] = torch.max(distances)
            else:
                # No matches - return zeros
                features["mean_distance"] = torch.tensor(0.0)
                features["std_distance"] = torch.tensor(0.0)
                features["median_distance"] = torch.tensor(0.0)
                features["min_distance"] = torch.tensor(0.0)
                features["max_distance"] = torch.tensor(0.0)

        if feature_types is None or "quality" in feature_types:
            # Match quality features
            if self["meta"]["n_matches"] > 0:
                quality = self["match_quality"]
                features["mean_quality"] = torch.mean(quality)
                features["std_quality"] = torch.std(quality)
                features["min_quality"] = torch.min(quality)
                features["max_quality"] = torch.max(quality)
            else:
                features["mean_quality"] = torch.tensor(0.0)
                features["std_quality"] = torch.tensor(0.0)
                features["min_quality"] = torch.tensor(0.0)
                features["max_quality"] = torch.tensor(0.0)

        return features

    def perform_crossmatch(self) -> CrossMatchTensorDict:
        """
        Führt Cross-Match zwischen den Katalogen durch.

        Returns:
            Self mit gefüllten Match-Informationen
        """
        algorithm = self["meta", "algorithm"]

        if algorithm == "nearest_neighbor":
            return self._nearest_neighbor_match()
        elif algorithm == "all_pairs":
            return self._all_pairs_match()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _nearest_neighbor_match(self) -> CrossMatchTensorDict:
        """Nächster-Nachbar-Matching."""
        cat1 = self["catalog1"]
        cat2 = self["catalog2"]
        match_radius = self["meta", "match_radius"]

        # Extrahiere Positionen
        if hasattr(cat1, "spatial"):
            pos1 = cat1.spatial.to_spherical()[:2]  # RA, Dec
        else:
            pos1 = cat1["spatial"].to_spherical()[:2]

        if hasattr(cat2, "spatial"):
            pos2 = cat2.spatial.to_spherical()[:2]
        else:
            pos2 = cat2["spatial"].to_spherical()[:2]

        ra1, dec1 = pos1
        ra2, dec2 = pos2

        matches = []
        distances = []
        qualities = []

        # Für jeden Eintrag in Katalog 1, finde nächsten Nachbarn in Katalog 2
        for i in range(len(ra1)):
            # Berechne Winkeldistanzen zu allen Objekten in Katalog 2
            sep = self._angular_separation(ra1[i], dec1[i], ra2, dec2)

            # Finde nächsten Nachbarn
            min_sep, min_idx = torch.min(sep, dim=0)

            # Prüfe ob innerhalb des Such-Radius
            if min_sep <= match_radius / 3600:  # Convert arcsec to degrees
                matches.append([i, min_idx.item()])
                distances.append(min_sep.item() * 3600)  # Convert to arcsec
                qualities.append(1.0 / (1.0 + min_sep.item()))  # quality metric

        if matches:
            self["matches"] = torch.tensor(matches, dtype=torch.long)
            self["distances"] = torch.tensor(distances)
            self["match_quality"] = torch.tensor(qualities)
        else:
            self["matches"] = torch.empty((0, 2), dtype=torch.long)
            self["distances"] = torch.empty(0)
            self["match_quality"] = torch.empty(0)

        self["meta", "n_matches"] = len(matches)
        self._compute_match_statistics()

        return self

    def _all_pairs_match(self) -> CrossMatchTensorDict:
        """All-pairs matching (all matches within radius) - vectorized."""
        cat1 = self["catalog1"]
        cat2 = self["catalog2"]
        match_radius = self["meta", "match_radius"]

        # Extrahiere Positionen
        if hasattr(cat1, "spatial"):
            pos1 = cat1.spatial.to_spherical()[:2]
        else:
            pos1 = cat1["spatial"].to_spherical()[:2]

        if hasattr(cat2, "spatial"):
            pos2 = cat2.spatial.to_spherical()[:2]
        else:
            pos2 = cat2["spatial"].to_spherical()[:2]

        ra1, dec1 = pos1
        ra2, dec2 = pos2

        # Vectorized angular separation computation
        # Expand dimensions for broadcasting: [N1, 1] and [1, N2]
        ra1_exp = ra1.unsqueeze(1)  # [N1, 1]
        dec1_exp = dec1.unsqueeze(1)  # [N1, 1]
        ra2_exp = ra2.unsqueeze(0)  # [1, N2]
        dec2_exp = dec2.unsqueeze(0)  # [1, N2]

        # Compute all pairwise angular separations at once
        # Result shape: [N1, N2]
        separations = self._angular_separation(ra1_exp, dec1_exp, ra2_exp, dec2_exp)

        # Find all matches within radius
        match_radius_deg = match_radius / 3600  # Convert arcsec to degrees
        mask = separations <= match_radius_deg

        # Get indices of matches
        indices = torch.nonzero(
            mask, as_tuple=False
        )  # [M, 2] where M is number of matches

        if indices.numel() > 0:
            # Extract matched pairs and their properties
            matches = indices  # Already in [i, j] format
            distances = separations[mask] * 3600  # Convert back to arcsec
            qualities = 1.0 / (1.0 + separations[mask])

            self["matches"] = matches
            self["distances"] = distances
            self["match_quality"] = qualities
        else:
            self["matches"] = torch.empty((0, 2), dtype=torch.long)
            self["distances"] = torch.empty(0)
            self["match_quality"] = torch.empty(0)

        self["meta", "n_matches"] = indices.shape[0] if indices.numel() > 0 else 0
        self._compute_match_statistics()

        return self

    def _angular_separation(
        self,
        ra1: torch.Tensor,
        dec1: torch.Tensor,
        ra2: torch.Tensor,
        dec2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute angular separation between celestial coordinates.

        Supports both scalar and vectorized operations via broadcasting.

        Args:
            ra1, dec1: First coordinates in degrees (can be broadcasted)
            ra2, dec2: Second coordinates in degrees (can be broadcasted)

        Returns:
            Angular separation in degrees (supports broadcasting)
        """
        # Convert to radians
        ra1_rad = ra1 * math.pi / 180
        dec1_rad = dec1 * math.pi / 180
        ra2_rad = ra2 * math.pi / 180
        dec2_rad = dec2 * math.pi / 180

        # Spherical trigonometry with broadcasting support
        cos_sep = torch.sin(dec1_rad) * torch.sin(dec2_rad) + torch.cos(
            dec1_rad
        ) * torch.cos(dec2_rad) * torch.cos(ra1_rad - ra2_rad)

        # Numerical stability
        cos_sep = torch.clamp(cos_sep, -1.0, 1.0)

        return torch.acos(cos_sep) * 180 / math.pi

    def _compute_match_statistics(self):
        """Berechnet Match-Statistiken."""
        n_matches = self["meta", "n_matches"]
        n_cat1 = self["catalog1"].n_objects
        self["catalog2"].n_objects

        if n_matches > 0:
            distances = self["distances"]
            stats = {
                "completeness": n_matches / n_cat1,
                "contamination": 0.0,  # Würde externe Validierung benötigen
                "mean_distance": torch.mean(distances).item(),
                "median_distance": torch.median(distances).item(),
                "std_distance": torch.std(distances).item(),
                "max_distance": torch.max(distances).item(),
                "min_distance": torch.min(distances).item(),
            }
        else:
            stats = {
                "completeness": 0.0,
                "contamination": 0.0,
                "mean_distance": 0.0,
                "median_distance": 0.0,
                "std_distance": 0.0,
                "max_distance": 0.0,
                "min_distance": 0.0,
            }

        self["meta", "match_statistics"] = stats

    def get_matched_catalog1(self) -> AstroTensorDict:
        """
        Extrahiert gematchte Objekte aus Katalog 1.

        Returns:
            AstroTensorDict mit gematchten Objekten
        """
        if self["meta", "n_matches"] == 0:
            raise ValueError("No matches found. Run perform_crossmatch() first.")

        indices = self["matches"][:, 0]

        # Extrahiere relevante Daten
        matched_data = {}
        cat1 = self["catalog1"]

        for key, value in cat1.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == cat1.n_objects:
                matched_data[key] = value[indices]
            else:
                matched_data[key] = value  # Metadaten bleiben unverändert

        return AstroTensorDict(matched_data)

    def get_matched_catalog2(self) -> AstroTensorDict:
        """
        Extrahiert gematchte Objekte aus Katalog 2.

        Returns:
            AstroTensorDict mit gematchten Objekten
        """
        if self["meta", "n_matches"] == 0:
            raise ValueError("No matches found. Run perform_crossmatch() first.")

        indices = self["matches"][:, 1]

        # Extrahiere relevante Daten
        matched_data = {}
        cat2 = self["catalog2"]

        for key, value in cat2.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == cat2.n_objects:
                matched_data[key] = value[indices]
            else:
                matched_data[key] = value

        return AstroTensorDict(matched_data)

    def get_unmatched_catalog1(self) -> AstroTensorDict:
        """
        Extrahiert nicht-gematchte Objekte aus Katalog 1.

        Returns:
            AstroTensorDict mit nicht-gematchten Objekten
        """
        if self["meta", "n_matches"] == 0:
            return self["catalog1"]  # Alle sind unmatched

        matched_indices = self["matches"][:, 0]
        all_indices = torch.arange(self["catalog1"].n_objects)
        unmatched_mask = ~torch.isin(all_indices, matched_indices)
        unmatched_indices = all_indices[unmatched_mask]

        # Extrahiere relevante Daten
        unmatched_data = {}
        cat1 = self["catalog1"]

        for key, value in cat1.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == cat1.n_objects:
                unmatched_data[key] = value[unmatched_indices]
            else:
                unmatched_data[key] = value

        return AstroTensorDict(unmatched_data)

    def print_statistics(self):
        """Druckt Match-Statistiken."""
        stats = self["meta", "match_statistics"]
        n_matches = self["meta", "n_matches"]
        n_cat1 = self["catalog1"].n_objects
        n_cat2 = self["catalog2"].n_objects

        print("Cross-Match Statistiken:")
        print(f"  Katalog 1: {n_cat1} Objekte")
        print(f"  Katalog 2: {n_cat2} Objekte")
        print(f"  Matches: {n_matches}")
        print(f"  Vollständigkeit: {stats['completeness']:.3f}")
        print(f"  Mittlere Distanz: {stats['mean_distance']:.2f} arcsec")
        print(f"  Median Distanz: {stats['median_distance']:.2f} arcsec")
        print(f"  Std Distanz: {stats['std_distance']:.2f} arcsec")
        print(
            f"  Min/Max Distanz: {stats['min_distance']:.2f}/"
            f"{stats['max_distance']:.2f} arcsec"
        )
