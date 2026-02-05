"""
Image TensorDict for astronomical image data.

TensorDict for astronomical image data with WCS and photometry support.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from .base import AstroTensorDict
from .mixins import FeatureExtractionMixin, NormalizationMixin, ValidationMixin


class ImageTensorDict(
    AstroTensorDict, NormalizationMixin, FeatureExtractionMixin, ValidationMixin
):
    """
    TensorDict for astronomical image data.

    Features:
    - WCS (World Coordinate System) handling
    - Source detection and aperture photometry
    - Background estimation and subtraction
    - PSF analysis and modeling
    - Multi-band image processing
    """

    def __init__(
        self,
        images: torch.Tensor,
        wcs: Optional[Union[WCS, List[WCS]]] = None,
        bands: Optional[List[str]] = None,
        pixel_scale: Optional[Union[float, List[float]]] = None,
        exposure_time: Optional[Union[float, torch.Tensor]] = None,
        zero_point: Optional[Union[float, torch.Tensor]] = None,
        image_type: str = "science",
        coordinates: Optional[SkyCoord] = None,
        **kwargs,
    ):
        """
        Initialize ImageTensorDict.

        Args:
            images: [N, C, H, W] Tensor with images (N=objects, C=bands, H/W=spatial)
            wcs: World Coordinate System(s) for astrometric calibration
            bands: Photometric band names
            pixel_scale: Pixel scale in arcsec/pixel
            exposure_time: Exposure time(s) in seconds
            zero_point: Photometric zero point(s)
            image_type: Type of image ('science', 'calibration', 'reference')
            coordinates: Central coordinates of images
        """
        if images.dim() != 4:
            raise ValueError(
                f"Images must be 4D tensor [N, C, H, W], got {images.shape}"
            )

        N, C, H, W = images.shape

        # Default bands if not provided
        if bands is None:
            bands = [f"band_{i}" for i in range(C)]
        elif len(bands) != C:
            raise ValueError(
                f"Number of bands ({len(bands)}) doesn't match channels ({C})"
            )

        # Handle WCS
        if wcs is not None:
            if isinstance(wcs, WCS):
                wcs = [wcs] * N  # Same WCS for all images
            elif len(wcs) != N:
                raise ValueError(
                    f"Number of WCS ({len(wcs)}) doesn't match images ({N})"
                )

        # Handle pixel scale
        if pixel_scale is not None:
            if isinstance(pixel_scale, (int, float)):
                pixel_scale = [pixel_scale] * N
            elif len(pixel_scale) != N:
                raise ValueError("Number of pixel scales doesn't match images")

        data = {
            "images": images,
            "meta": {
                "n_objects": N,
                "n_bands": C,
                "image_shape": (H, W),
                "bands": bands,
                "image_type": image_type,
                "pixel_scale": pixel_scale,
                "wcs_available": wcs is not None,
            },
        }

        if wcs is not None:
            data["wcs"] = wcs

        if exposure_time is not None:
            if isinstance(exposure_time, (int, float)):
                exposure_time = torch.tensor(exposure_time, dtype=torch.float32)
            data["exposure_time"] = exposure_time

        if zero_point is not None:
            if isinstance(zero_point, (int, float)):
                zero_point = torch.tensor(zero_point, dtype=torch.float32)
            data["zero_point"] = zero_point

        if coordinates is not None:
            data["coordinates"] = coordinates

        super().__init__(data, batch_size=(N,), **kwargs)

    @property
    def images(self) -> torch.Tensor:
        """Image data [N, C, H, W]."""
        return self["images"]

    @property
    def image_shape(self) -> Tuple[int, int]:
        """Image spatial dimensions (H, W)."""
        return self._metadata["image_shape"]

    @property
    def n_bands(self) -> int:
        """Number of photometric bands."""
        return self._metadata["n_bands"]

    @property
    def bands(self) -> List[str]:
        """Photometric band names."""
        return self._metadata["bands"]

    @property
    def image_type(self) -> str:
        """Image type identifier."""
        return self._metadata["image_type"]

    @property
    def pixel_scale(self) -> Optional[List[float]]:
        """Pixel scales in arcsec/pixel."""
        return self._metadata["pixel_scale"]

    @property
    def wcs_list(self) -> Optional[List[WCS]]:
        """List of WCS objects."""
        return self.get("wcs", None)

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract image features from the TensorDict.

        Args:
            feature_types: Types of features to extract ('image', 'morphological', 'statistical')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted image features
        """
        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Add image-specific computed features
        if feature_types is None or "image" in feature_types:
            # Basic image statistics
            images = self.images
            features["mean_intensity"] = torch.mean(images, dim=(-2, -1))
            features["std_intensity"] = torch.std(images, dim=(-2, -1))
            features["max_intensity"] = torch.max(
                images.view(images.shape[0], images.shape[1], -1), dim=-1
            )[0]
            features["min_intensity"] = torch.min(
                images.view(images.shape[0], images.shape[1], -1), dim=-1
            )[0]

        if feature_types is None or "morphological" in feature_types:
            # Morphological features
            images = self.images

            # Gradient-based features
            grad_x = torch.diff(images, dim=-1)
            grad_y = torch.diff(images, dim=-2)

            features["gradient_strength_x"] = torch.mean(
                torch.abs(grad_x), dim=(-2, -1)
            )
            features["gradient_strength_y"] = torch.mean(
                torch.abs(grad_y), dim=(-2, -1)
            )

        if feature_types is None or "statistical" in feature_types:
            # Statistical features
            images = self.images

            # Flatten spatial dimensions for percentile calculations
            flat_images = images.view(images.shape[0], images.shape[1], -1)

            features["median_intensity"] = torch.median(flat_images, dim=-1)[0]
            features["intensity_range"] = (
                features["max_intensity"] - features["min_intensity"]
            )

        return features

    def get_wcs(self, image_index: int = 0) -> Optional[WCS]:
        """Get WCS for specific image."""
        if "wcs" not in self:
            return None
        return self["wcs"][image_index]

    def pixel_to_sky(
        self,
        x_pix: Union[float, np.ndarray],
        y_pix: Union[float, np.ndarray],
        image_index: int = 0,
    ) -> Optional[SkyCoord]:
        """
        Convert pixel coordinates to sky coordinates.

        Args:
            x_pix: X pixel coordinate(s)
            y_pix: Y pixel coordinate(s)
            image_index: Which image's WCS to use

        Returns:
            Sky coordinates or None if no WCS available
        """
        wcs = self.get_wcs(image_index)
        if wcs is None:
            return None

        sky_coords = wcs.pixel_to_world(x_pix, y_pix)
        return sky_coords

    def sky_to_pixel(
        self, coordinates: SkyCoord, image_index: int = 0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert sky coordinates to pixel coordinates.

        Args:
            coordinates: Sky coordinates
            image_index: Which image's WCS to use

        Returns:
            Tuple of (x_pix, y_pix) arrays or None if no WCS
        """
        wcs = self.get_wcs(image_index)
        if wcs is None:
            return None

        x_pix, y_pix = wcs.world_to_pixel(coordinates)
        return x_pix, y_pix

    def estimate_background(self, method: str = "sigma_clip") -> torch.Tensor:
        """
        Estimate image background using statistical methods.

        Args:
            method: Background estimation method ('sigma_clip', 'median', 'mode')

        Returns:
            Background estimate [N, C] - one value per image/band
        """
        backgrounds = torch.zeros(self.n_objects, self.n_bands)

        for n in range(self.n_objects):
            for c in range(self.n_bands):
                image = self.images[n, c].detach().cpu().numpy()

                if method == "sigma_clip":
                    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
                    background = median
                elif method == "median":
                    background = np.median(image)
                elif method == "mode":
                    # Approximate mode as 3*median - 2*mean
                    background = 3 * np.median(image) - 2 * np.mean(image)
                else:
                    background = np.mean(image)

                backgrounds[n, c] = background

        return backgrounds

    def subtract_background(
        self, method: str = "sigma_clip", **kwargs
    ) -> "ImageTensorDict":
        """
        Subtract background from images.

        Args:
            method: Background estimation method
            **kwargs: Additional arguments for background estimation
        """
        backgrounds = self.estimate_background(method=method)

        # Broadcast background to image shape
        background_map = backgrounds.unsqueeze(-1).unsqueeze(-1)
        background_subtracted = self.images - background_map

        result = ImageTensorDict(
            background_subtracted,
            self.wcs_list,
            self.bands,
            pixel_scale=self.pixel_scale,
            exposure_time=self.get("exposure_time", None),
            zero_point=self.get("zero_point", None),
            image_type=self.image_type,
            coordinates=self.get("coordinates", None),
        )
        result.add_history("subtract_background", method=method)
        return result

    def detect_sources_simple(
        self, threshold: float = 5.0, band_index: int = 0
    ) -> List[Dict]:
        """
        Simple source detection using threshold.

        Args:
            threshold: Detection threshold in sigma above background
            band_index: Which band to use for detection

        Returns:
            List of source catalogs (one per image)
        """
        source_catalogs = []

        for n in range(self.n_objects):
            image = self.images[n, band_index].detach().cpu().numpy()

            # Estimate background statistics
            mean, median, std = sigma_clipped_stats(image, sigma=3.0)

            # Simple threshold detection
            detection_mask = image > (median + threshold * std)

            if np.any(detection_mask):
                # Find connected components (simplified)
                y_coords, x_coords = np.where(detection_mask)
                catalog = {
                    "x": x_coords.astype(float),
                    "y": y_coords.astype(float),
                    "flux": image[detection_mask],
                    "n_sources": len(x_coords),
                }
            else:
                catalog = {
                    "x": np.array([]),
                    "y": np.array([]),
                    "flux": np.array([]),
                    "n_sources": 0,
                }

            source_catalogs.append(catalog)

        return source_catalogs

    def crop_center(self, crop_size: Tuple[int, int]) -> "ImageTensorDict":
        """Crop center region of images."""
        N, C, H, W = self.images.shape
        crop_h, crop_w = crop_size

        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2

        cropped = self.images[
            :, :, start_h : start_h + crop_h, start_w : start_w + crop_w
        ]

        result = ImageTensorDict(
            cropped,
            self.wcs_list,
            self.bands,
            pixel_scale=self.pixel_scale,
            exposure_time=self.get("exposure_time", None),
            zero_point=self.get("zero_point", None),
            image_type=self.image_type,
            coordinates=self.get("coordinates", None),
        )
        result.add_history("crop_center", crop_size=crop_size)
        return result

    def resize(self, target_size: Tuple[int, int]) -> "ImageTensorDict":
        """Resize images to target size."""
        import torch.nn.functional as F

        images = F.interpolate(
            self.images, size=target_size, mode="bilinear", align_corners=False
        )

        result = ImageTensorDict(
            images,
            self.wcs_list,
            self.bands,
            pixel_scale=self.pixel_scale,
            exposure_time=self.get("exposure_time", None),
            zero_point=self.get("zero_point", None),
            image_type=self.image_type,
            coordinates=self.get("coordinates", None),
        )
        result.add_history("resize", target_size=target_size)
        return result

    def extract_image_features(self) -> torch.Tensor:
        """
        Extract comprehensive image features for analysis.

        Returns:
            [N, F] Feature tensor with image properties
        """
        features = []

        for n in range(self.n_objects):
            img_features = []

            for c in range(self.n_bands):
                image = self.images[n, c]

                # Basic statistics
                img_features.extend(
                    [
                        torch.mean(image),
                        torch.std(image),
                        torch.median(image),
                        torch.min(image),
                        torch.max(image),
                    ]
                )

                # Image structure
                gradient_x = torch.diff(image, dim=1)
                gradient_y = torch.diff(image, dim=0)

                img_features.extend(
                    [
                        torch.mean(torch.abs(gradient_x)),  # Edge strength X
                        torch.mean(torch.abs(gradient_y)),  # Edge strength Y
                        torch.std(gradient_x),  # Texture X
                        torch.std(gradient_y),  # Texture Y
                    ]
                )

            features.append(torch.stack(img_features))

        return torch.stack(features)

    def validate(self) -> bool:
        """Validate image tensor data."""
        if not super().validate():
            return False

        return (
            "images" in self
            and self.images.dim() == 4
            and self.images.shape[1] == len(self.bands)
            and self.images.shape[2] > 10  # Minimum image size
            and self.images.shape[3] > 10
            and self.validate_finite_values("images")
        )
