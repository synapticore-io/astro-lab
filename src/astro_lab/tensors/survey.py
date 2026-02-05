"""
Survey TensorDict for multi-component astronomical survey data.

TensorDict for survey data containing multiple components
(spatial, photometric, images, etc.) with unified feature extraction.
"""

from typing import Any, Dict, List, Optional

import torch

from .base import AstroTensorDict
from .image import ImageTensorDict
from .mixins import ValidationMixin
from .photometric import PhotometricTensorDict
from .spatial import SpatialTensorDict


class SurveyTensorDict(AstroTensorDict, ValidationMixin):
    """
    TensorDict for multi-component survey data.

    Structure:
    {
        "spatial": SpatialTensorDict,      # 3D coordinates
        "photometric": PhotometricTensorDict,  # Multi-band photometry
        "image": ImageTensorDict,          # Image data (optional)
        "survey_name": str,                # Survey identifier
        "meta": Dict[str, Any],            # Additional metadata
    }
    """

    def __init__(
        self,
        spatial: Optional[SpatialTensorDict] = None,
        photometric: Optional[PhotometricTensorDict] = None,
        image: Optional[ImageTensorDict] = None,
        survey_name: str = "",
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize SurveyTensorDict.

        Args:
            spatial: Spatial coordinates
            photometric: Multi-band photometry
            image: Image data (optional)
            survey_name: Survey identifier
            meta: Additional metadata
        """
        data = {}

        if spatial is not None:
            data["spatial"] = spatial

        if photometric is not None:
            data["photometric"] = photometric

        if image is not None:
            data["image"] = image

        if survey_name:
            data["survey_name"] = survey_name

        if meta is not None:
            data["meta"] = meta

        # Determine batch size from components
        batch_size = None
        if spatial is not None:
            batch_size = spatial.batch_size
        elif photometric is not None:
            batch_size = photometric.batch_size
        elif image is not None:
            batch_size = image.batch_size

        super().__init__(data, batch_size=batch_size, **kwargs)

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract survey features from all available components.

        Args:
            feature_types: Types of features to extract ('survey', 'spatial', 'photometric', 'image')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted features from all components
        """
        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Extract features from each component
        if feature_types is None or "spatial" in feature_types:
            if self.has_spatial():
                spatial_features = self.spatial.extract_features(["spatial"], **kwargs)
                # Prefix with component name to avoid conflicts
                for key, value in spatial_features.items():
                    features[f"spatial_{key}"] = value

        if feature_types is None or "photometric" in feature_types:
            if self.has_photometric():
                photo_features = self.photometric.extract_features(
                    ["photometric"], **kwargs
                )
                for key, value in photo_features.items():
                    features[f"photometric_{key}"] = value

        if feature_types is None or "image" in feature_types:
            if self.has_image():
                image_features = self.image.extract_features(["image"], **kwargs)
                for key, value in image_features.items():
                    features[f"image_{key}"] = value

        if feature_types is None or "survey" in feature_types:
            # Survey-level aggregated features
            features["survey_name"] = torch.tensor(
                hash(self.survey_name) % 10000, dtype=torch.float32
            )
            features["has_spatial"] = torch.tensor(float(self.has_spatial()))
            features["has_photometric"] = torch.tensor(float(self.has_photometric()))
            features["has_image"] = torch.tensor(float(self.has_image()))

            # Component count
            features["n_components"] = torch.tensor(
                float(self.has_spatial() + self.has_photometric() + self.has_image())
            )

        return features

    @property
    def spatial(self) -> Optional[SpatialTensorDict]:
        """Spatial coordinates."""
        return self.get("spatial", None)

    @property
    def photometric(self) -> Optional[PhotometricTensorDict]:
        """Multi-band photometry."""
        return self.get("photometric", None)

    @property
    def image(self) -> Optional[ImageTensorDict]:
        """Image data."""
        return self.get("image", None)

    @property
    def survey_name(self) -> str:
        """Survey identifier."""
        return self.get("survey_name", "")

    @property
    def meta(self) -> Dict[str, Any]:
        """Additional metadata."""
        return self.get("meta", {})

    def has_spatial(self) -> bool:
        """Check if spatial data is available."""
        return "spatial" in self

    def has_photometric(self) -> bool:
        """Check if photometric data is available."""
        return "photometric" in self

    def has_image(self) -> bool:
        """Check if image data is available."""
        return "image" in self

    def get_coordinates(self) -> Optional[torch.Tensor]:
        """Get spatial coordinates if available."""
        if self.has_spatial():
            return self.spatial.coordinates
        return None

    def get_magnitudes(self) -> Optional[torch.Tensor]:
        """Get photometric magnitudes if available."""
        if self.has_photometric():
            return self.photometric.magnitudes
        return None

    def get_bands(self) -> Optional[list]:
        """Get photometric bands if available."""
        if self.has_photometric():
            return self.photometric.bands
        return None

    def validate(self) -> bool:
        """
        Validate that the survey tensor has at least one data component.

        Returns:
            True if valid, False otherwise
        """
        return any([self.has_spatial(), self.has_photometric(), self.has_image()])

    def to_cartesian(self) -> Optional[torch.Tensor]:
        """Convert to Cartesian coordinates if spatial data is available."""
        if self.has_spatial():
            return self.spatial.to_cartesian()
        return None

    def to_flux(self) -> "SurveyTensorDict":
        """Convert photometric data to flux units."""
        if self.has_photometric():
            new_photometric = self.photometric.to_flux()
            return SurveyTensorDict(
                spatial=self.spatial,
                photometric=new_photometric,
                image=self.image,
                survey_name=self.survey_name,
                meta=self.meta,
            )
        return self

    def to_magnitude(self) -> "SurveyTensorDict":
        """Convert photometric data to magnitude units."""
        if self.has_photometric():
            new_photometric = self.photometric.to_magnitude()
            return SurveyTensorDict(
                spatial=self.spatial,
                photometric=new_photometric,
                image=self.image,
                survey_name=self.survey_name,
                meta=self.meta,
            )
        return self

    def __repr__(self) -> str:
        """String representation."""
        components = []
        if self.has_spatial():
            components.append("spatial")
        if self.has_photometric():
            components.append("photometric")
        if self.has_image():
            components.append("image")

        survey_info = f"'{self.survey_name}'" if self.survey_name else "unnamed"
        return f"SurveyTensorDict({survey_info}, components: {components})"
