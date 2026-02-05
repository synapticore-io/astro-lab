"""
Lightcurve TensorDict for time series photometry data.

TensorDict for time series photometry data with astronomical analysis.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.timeseries import LombScargle

from .base import AstroTensorDict
from .mixins import FeatureExtractionMixin, NormalizationMixin, ValidationMixin


class LightcurveTensorDict(
    AstroTensorDict, NormalizationMixin, FeatureExtractionMixin, ValidationMixin
):
    """
    TensorDict for lightcurve data with astronomical time series analysis.

    Features:
    - Time handling with astropy.time
    - Period detection using Lomb-Scargle periodograms
    - Variability classification and characterization
    - Phase folding and lightcurve modeling
    - Multi-band lightcurve analysis
    - Outlier detection and data quality assessment
    """

    # Common variable star periods for validation (days)
    VARIABLE_STAR_PERIODS = {
        "RR_Lyrae": (0.2, 1.0),
        "Cepheid": (1.0, 100.0),
        "Delta_Scuti": (0.01, 0.3),
        "Beta_Lyrae": (0.5, 50.0),
        "W_UMa": (0.2, 1.0),
        "Mira": (80, 1000),
        "Semiregular": (20, 2000),
    }

    def __init__(
        self,
        times: torch.Tensor,
        magnitudes: torch.Tensor,
        magnitude_errors: Optional[torch.Tensor] = None,
        bands: Optional[List[str]] = None,
        time_format: str = "mjd",
        time_scale: str = "utc",
        filter_system: str = "AB",
        coordinates: Optional[SkyCoord] = None,
        object_ids: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize LightcurveTensorDict.

        Args:
            times: [N, T] Time points (MJD, JD, etc.)
            magnitudes: [N, T] or [N, T, B] Magnitude/flux values
            magnitude_errors: [N, T] or [N, T, B] Uncertainties (optional)
            bands: List of filter bands (required for multi-band data)
            time_format: Time format ('mjd', 'jd', 'isot', etc.)
            time_scale: Time scale ('utc', 'tdb', 'tai', etc.)
            filter_system: Magnitude system ('AB', 'Vega', 'ST')
            coordinates: Source coordinates
            object_ids: Object identifiers for each lightcurve
        """
        if times.shape[:2] != magnitudes.shape[:2]:
            raise ValueError(
                f"Times and magnitudes must have compatible shapes, "
                f"got {times.shape} and {magnitudes.shape}"
            )
        if magnitude_errors is not None and magnitude_errors.shape != magnitudes.shape:
            raise ValueError(
                f"Errors must have same shape as magnitudes, got "
                f"{magnitude_errors.shape} and {magnitudes.shape}"
            )

        n_objects, n_times = times.shape

        # Handle multi-band vs single-band data
        if magnitudes.dim() == 3:
            # Multi-band: [N, T, B]
            n_bands = magnitudes.shape[2]
            if bands is None:
                bands = [f"band_{i}" for i in range(n_bands)]
            elif len(bands) != n_bands:
                raise ValueError(
                    f"Number of bands ({len(bands)}) doesn't match data ({n_bands})"
                )
        else:
            # Single band: [N, T]
            n_bands = 1
            if bands is None:
                bands = ["V"]  # Default to V band
            magnitudes = magnitudes.unsqueeze(-1)  # Add band dimension
            if magnitude_errors is not None:
                magnitude_errors = magnitude_errors.unsqueeze(-1)

        data = {
            "times": times,
            "magnitudes": magnitudes,
            "meta": {
                "n_objects": n_objects,
                "n_times": n_times,
                "n_bands": n_bands,
                "bands": bands,
                "time_format": time_format,
                "time_scale": time_scale,
                "filter_system": filter_system,
                "object_ids": object_ids or [f"object_{i}" for i in range(n_objects)],
                "baseline": (times.min().item(), times.max().item()),
            },
        }

        if magnitude_errors is not None:
            data["magnitude_errors"] = magnitude_errors

        if coordinates is not None:
            data["coordinates"] = coordinates

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def times(self) -> torch.Tensor:
        """Time points [N, T]."""
        return self["times"]

    @property
    def magnitudes(self) -> torch.Tensor:
        """Magnitude/flux values [N, T, B]."""
        return self["magnitudes"]

    @property
    def magnitude_errors(self) -> Optional[torch.Tensor]:
        """Magnitude/flux uncertainties [N, T, B]."""
        return self.get("magnitude_errors", None)

    @property
    def bands(self) -> List[str]:
        """Photometric bands."""
        return self._metadata["bands"]

    @property
    def n_bands(self) -> int:
        """Number of photometric bands."""
        return self._metadata["n_bands"]

    @property
    def baseline(self) -> Tuple[float, float]:
        """Time baseline (start, end)."""
        return self._metadata["baseline"]

    @property
    def object_ids(self) -> List[str]:
        """Object identifiers."""
        return self._metadata["object_ids"]

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract lightcurve features from the TensorDict.

        Args:
            feature_types: Types of features to extract ('temporal', 'variability', 'periodic')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted lightcurve features
        """
        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Add lightcurve-specific computed features
        if feature_types is None or "temporal" in feature_types:
            # Basic temporal features
            times = self.times
            features["baseline"] = times[:, -1] - times[:, 0]
            features["n_observations"] = torch.full(
                (self.n_objects,), self.times.shape[1], dtype=torch.float32
            )

            # Cadence features
            time_diffs = torch.diff(times, dim=1)
            features["median_cadence"] = torch.median(time_diffs, dim=1)[0]
            features["cadence_std"] = torch.std(time_diffs, dim=1)

        if feature_types is None or "variability" in feature_types:
            # Variability features for first band
            mags = self.magnitudes[:, :, 0]  # Use first band

            features["mean_magnitude"] = torch.mean(mags, dim=1)
            features["magnitude_std"] = torch.std(mags, dim=1)
            features["magnitude_range"] = (
                torch.max(mags, dim=1)[0] - torch.min(mags, dim=1)[0]
            )
            features["amplitude"] = features["magnitude_range"] / 2.0

            # Coefficient of variation
            features["coefficient_variation"] = features["magnitude_std"] / (
                features["mean_magnitude"] + 1e-10
            )

            # Stetson variability index (simplified)
            if self.magnitude_errors is not None:
                errors = self.magnitude_errors[:, :, 0]
                weighted_residuals = (
                    mags - features["mean_magnitude"].unsqueeze(1)
                ) / (errors + 1e-10)
                features["stetson_index"] = (
                    torch.mean(weighted_residuals**2, dim=1) - 1.0
                )

        if feature_types is None or "periodic" in feature_types:
            # Add periodic features if not already computed
            if "best_period" not in features:
                period_info = self.find_best_periods(n_periods=1)
                features["best_period"] = period_info["periods"][:, 0, 0]
                features["period_power"] = period_info["powers"][:, 0, 0]
                features["period_significance"] = period_info["significance"][:, 0]

        return features

    def to_astropy_time(self, object_index: int = 0) -> Time:
        """
        Convert times to astropy Time object.

        Args:
            object_index: Which object's times to convert
        """
        times_array = self.times[object_index].detach().cpu().numpy()
        return Time(
            times_array,
            format=self._metadata["time_format"],
            scale=self._metadata["time_scale"],
        )

    def compute_lomb_scargle_periodogram(
        self,
        object_index: int = 0,
        band_index: int = 0,
        min_period: float = 0.1,
        max_period: Optional[float] = None,
        samples_per_peak: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Lomb-Scargle periodogram for period detection.

        Args:
            object_index: Which object to analyze
            band_index: Which photometric band to use
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (if None, use baseline/3)
            samples_per_peak: Frequency sampling rate

        Returns:
            Tuple of (frequencies, power)
        """
        times = self.times[object_index].detach().cpu().numpy()
        mags = self.magnitudes[object_index, :, band_index].detach().cpu().numpy()

        # Handle errors if available
        if self.magnitude_errors is not None:
            errors = (
                self.magnitude_errors[object_index, :, band_index]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            errors = None

        # Filter out NaN/invalid values
        valid_mask = np.isfinite(times) & np.isfinite(mags)
        if errors is not None:
            valid_mask &= np.isfinite(errors) & (errors > 0)

        times = times[valid_mask]
        mags = mags[valid_mask]
        if errors is not None:
            errors = errors[valid_mask]

        if len(times) < 10:
            # Not enough data points
            return torch.empty(0), torch.empty(0)

        # Set up frequency grid
        if max_period is None:
            max_period = (times.max() - times.min()) / 3.0

        min_frequency = 1.0 / max_period
        max_frequency = 1.0 / min_period

        # Use astropy's LombScargle
        ls = LombScargle(times, mags, dy=errors)
        frequency, power = ls.autopower(
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
            samples_per_peak=samples_per_peak,
        )

        return torch.tensor(frequency, dtype=torch.float32), torch.tensor(
            power, dtype=torch.float32
        )

    def find_best_periods(
        self,
        min_period: float = 0.1,
        max_period: Optional[float] = None,
        n_periods: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Find best periods for all objects and bands.

        Args:
            min_period: Minimum period to search (days)
            max_period: Maximum period to search
            n_periods: Number of top periods to return

        Returns:
            Dictionary with periods and powers for each object/band
        """
        results = {
            "periods": torch.zeros(self.n_objects, self.n_bands, n_periods),
            "powers": torch.zeros(self.n_objects, self.n_bands, n_periods),
            "significance": torch.zeros(self.n_objects, self.n_bands),
        }

        for obj_idx in range(self.n_objects):
            for band_idx in range(self.n_bands):
                frequency, power = self.compute_lomb_scargle_periodogram(
                    obj_idx, band_idx, min_period, max_period
                )

                if len(power) == 0:
                    continue

                # Find peaks
                peak_indices = torch.argsort(power, descending=True)[:n_periods]

                best_periods = 1.0 / frequency[peak_indices]
                best_powers = power[peak_indices]

                # Store results
                n_found = min(len(best_periods), n_periods)
                results["periods"][obj_idx, band_idx, :n_found] = best_periods[:n_found]
                results["powers"][obj_idx, band_idx, :n_found] = best_powers[:n_found]

                # Estimate significance (simple approach)
                if len(power) > 0:
                    results["significance"][obj_idx, band_idx] = (
                        power.max() / power.mean()
                    )

        return results

    def phase_fold(
        self,
        periods: Union[float, torch.Tensor],
        epoch: Optional[Union[float, torch.Tensor]] = None,
    ) -> "LightcurveTensorDict":
        """
        Phase fold lightcurves using given periods.

        Args:
            periods: Period(s) for phase folding (days)
            epoch: Reference epoch (if None, use minimum time)

        Returns:
            New LightcurveTensorDict with phase-folded data
        """
        if isinstance(periods, (int, float)):
            periods = torch.full((self.n_objects,), periods, dtype=torch.float32)

        if epoch is None:
            epoch = self.times.min(dim=1)[0]  # Use minimum time for each object
        elif isinstance(epoch, (int, float)):
            epoch = torch.full((self.n_objects,), epoch, dtype=torch.float32)

        # Compute phases: φ = (t - t0) / P mod 1
        phases = torch.zeros_like(self.times)
        for i in range(self.n_objects):
            phases[i] = ((self.times[i] - epoch[i]) / periods[i]) % 1.0

        # Sort by phase for each object
        sorted_phases = torch.zeros_like(phases)
        sorted_magnitudes = torch.zeros_like(self.magnitudes)
        sorted_errors = None
        if self.magnitude_errors is not None:
            sorted_errors = torch.zeros_like(self.magnitude_errors)

        for i in range(self.n_objects):
            sort_indices = torch.argsort(phases[i])
            sorted_phases[i] = phases[i][sort_indices]
            sorted_magnitudes[i] = self.magnitudes[i][sort_indices]
            if self.magnitude_errors is not None:
                sorted_errors[i] = self.magnitude_errors[i][sort_indices]

        result = LightcurveTensorDict(
            sorted_phases,
            sorted_magnitudes,
            sorted_errors,
            bands=self.bands,
            time_format="phase",
            filter_system=self._metadata["filter_system"],
            coordinates=self.get("coordinates", None),
        )
        result.add_history("phase_fold", periods=periods.tolist())
        return result

    def detect_outliers(
        self, sigma_threshold: float = 3.0, method: str = "sigma_clip"
    ) -> torch.Tensor:
        """
        Detect outlier data points.

        Args:
            sigma_threshold: Sigma threshold for outlier detection
            method: Detection method ('sigma_clip', 'iqr', 'mad')

        Returns:
            Boolean mask [N, T] indicating outliers (True = outlier)
        """
        outlier_mask = torch.zeros_like(self.times, dtype=torch.bool)

        for obj_idx in range(self.n_objects):
            for band_idx in range(self.n_bands):
                mags = self.magnitudes[obj_idx, :, band_idx].detach().cpu().numpy()

                # Remove NaN values for analysis
                valid_mask = np.isfinite(mags)
                valid_mags = mags[valid_mask]

                if len(valid_mags) < 10:
                    continue

                if method == "sigma_clip":
                    mean, median, std = sigma_clipped_stats(
                        valid_mags, sigma=sigma_threshold, maxiters=3
                    )
                    outliers = np.abs(mags - median) > (sigma_threshold * std)

                elif method == "iqr":
                    q1, q3 = np.percentile(valid_mags, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = (mags < lower_bound) | (mags > upper_bound)

                elif method == "mad":
                    median = np.median(valid_mags)
                    mad = np.median(np.abs(valid_mags - median))
                    outliers = np.abs(mags - median) > (sigma_threshold * mad)

                else:
                    raise ValueError(f"Unknown outlier detection method: {method}")

                # Update mask (average across bands)
                outlier_mask[obj_idx] |= torch.tensor(outliers, dtype=torch.bool)

        return outlier_mask

    def extract_variability_features(self) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive variability features for classification.

        Returns:
            Dictionary of variability features
        """
        features = {}

        for band_idx, band in enumerate(self.bands):
            band_features = {}

            # Extract magnitudes for this band
            band_mags = self.magnitudes[:, :, band_idx]  # [N, T]

            # Basic statistics
            band_features["mean"] = torch.mean(band_mags, dim=1)
            band_features["std"] = torch.std(band_mags, dim=1)
            band_features["min"] = torch.min(band_mags, dim=1)[0]
            band_features["max"] = torch.max(band_mags, dim=1)[0]
            band_features["range"] = band_features["max"] - band_features["min"]

            # Variability amplitude
            band_features["amplitude"] = band_features["range"] / 2.0

            # Coefficient of variation
            band_features["cv"] = band_features["std"] / (band_features["mean"] + 1e-10)

            # Skewness and kurtosis (simplified)
            centered = band_mags - band_features["mean"].unsqueeze(1)
            normalized = centered / (band_features["std"].unsqueeze(1) + 1e-10)
            band_features["skewness"] = torch.mean(normalized**3, dim=1)
            band_features["kurtosis"] = torch.mean(normalized**4, dim=1) - 3.0

            # Stetson variability index
            if self.magnitude_errors is not None:
                errors = self.magnitude_errors[:, :, band_idx]
                weighted_residuals = (
                    band_mags - band_features["mean"].unsqueeze(1)
                ) / (errors + 1e-10)
                band_features["stetson_j"] = (
                    torch.mean(weighted_residuals**2, dim=1) - 1.0
                )

            # Time-based features
            time_diffs = torch.diff(self.times, dim=1)
            band_features["median_cadence"] = torch.median(time_diffs, dim=1)[0]
            band_features["baseline"] = self.times[:, -1] - self.times[:, 0]

            # Store band features
            for feat_name, feat_values in band_features.items():
                features[f"{band}_{feat_name}"] = feat_values

        return features

    def classify_variability(self) -> Dict[str, torch.Tensor]:
        """
        Basic variability classification based on features.

        Returns:
            Dictionary with classification results
        """
        features = self.extract_variability_features()

        # Use first band for classification
        band = self.bands[0]
        amplitude = features[f"{band}_amplitude"]
        period_info = self.find_best_periods(n_periods=1)
        best_periods = period_info["periods"][:, 0, 0]  # First band, first period
        best_powers = period_info["powers"][:, 0, 0]

        # Initialize classification
        var_class = torch.zeros(self.n_objects, dtype=torch.long)
        confidence = torch.zeros(self.n_objects)

        # Simple classification rules
        for i in range(self.n_objects):
            amp = amplitude[i].item()
            period = best_periods[i].item()
            power = best_powers[i].item()

            # High significance periodic variable
            if power > 10 and period > 0:
                if 0.2 <= period <= 1.0 and amp > 0.3:
                    var_class[i] = 1  # RR Lyrae candidate
                    confidence[i] = min(power / 10.0, 1.0)
                elif 1.0 <= period <= 100.0 and amp > 0.5:
                    var_class[i] = 2  # Cepheid candidate
                    confidence[i] = min(power / 10.0, 1.0)
                elif period < 0.3 and amp > 0.1:
                    var_class[i] = 3  # Delta Scuti candidate
                    confidence[i] = min(power / 10.0, 1.0)
                else:
                    var_class[i] = 4  # Other periodic
                    confidence[i] = min(power / 10.0, 1.0)
            elif amp > 0.2:
                var_class[i] = 5  # Irregular variable
                confidence[i] = amp
            # else: var_class[i] = 0 (non-variable)

        return {
            "variability_class": var_class,
            "confidence": confidence,
            "periods": best_periods,
            "amplitudes": amplitude,
            "features": features,
        }

    def compute_colors_vs_time(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compute color evolution if multi-band data available.

        Returns:
            Dictionary with color indices vs time
        """
        if self.n_bands < 2:
            return None

        colors = {}

        # Standard color combinations
        color_pairs = []
        bands_set = set(self.bands)

        # Common astronomical colors
        if "g" in bands_set and "r" in bands_set:
            color_pairs.append(("g", "r"))
        if "B" in bands_set and "V" in bands_set:
            color_pairs.append(("B", "V"))
        if "J" in bands_set and "H" in bands_set:
            color_pairs.append(("J", "H"))

        # Add any adjacent bands
        for i in range(len(self.bands) - 1):
            color_pairs.append((self.bands[i], self.bands[i + 1]))

        for band1, band2 in color_pairs:
            try:
                idx1 = self.bands.index(band1)
                idx2 = self.bands.index(band2)

                color_name = f"{band1}-{band2}"
                colors[color_name] = (
                    self.magnitudes[:, :, idx1] - self.magnitudes[:, :, idx2]
                )

                # Color variability
                colors[f"{color_name}_std"] = torch.std(colors[color_name], dim=1)

            except ValueError:
                continue

        return colors

    def validate(self) -> bool:
        """Validate lightcurve tensor data."""
        if not super().validate():
            return False

        return (
            "times" in self
            and "magnitudes" in self
            and self.times.shape[0] == self.magnitudes.shape[0]
            and self.times.shape[1] == self.magnitudes.shape[1]
            and self.times.shape[1] > 5  # Minimum time points
            and self.validate_finite_values("times")
            and len(self.bands) == self.magnitudes.shape[2]
        )
