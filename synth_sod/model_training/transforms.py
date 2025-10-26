from enum import Enum
import albumentations as A
import cv2


class TransformMode(Enum):
    REGULAR = "regular"
    TEST = "test"
    SYNTHETIC = "synthetic"


def get_transforms(image_size: int, mode: str) -> A.Compose:
    """Get albumentations transforms based on specified mode."""
    base_transform = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]

    if mode == TransformMode.TEST.value:
        return A.Compose(base_transform, additional_targets={'mask': 'mask'})

    geometric_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),
        A.Rotate(limit=15, p=0.2),
    ]

    color_transforms = [
        A.OneOf([
            A.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.2,
                hue=0.2,
                p=0.7
            ),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        ], p=0.5),
    ]

    noise_transforms = [
        A.OneOf([
            A.GaussNoise(std_range=(0.2, 0.44)),
            A.ISONoise(p=0.5),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5)
        ], p=0.3),
    ]

    synthetic_transforms = [
        A.OneOf([
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,    # Moderate increase
                hue=0.2,          # Moderate increase
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=25,    # Moderate increase from 15
                sat_shift_limit=35,    # Moderate increase from 20
                val_shift_limit=30,    # Moderate increase from 20
                p=0.4
            ),
            A.CLAHE(
                clip_limit=4.0,        # Reduced from 6.0
                tile_grid_size=(8, 8),
                p=0.2
            ),
        ], p=0.7),

        # Enhanced noise and artifacts
        A.OneOf([
            A.ISONoise(
                color_shift=(0.01, 0.03),    # Moderate increase
                intensity=(0.08, 0.3),       # Moderate increase
                p=0.4
            ),
            A.GaussNoise(
                std_range=(0.25, 0.6),      # Moderate increase
                p=0.4
            ),
            A.MultiplicativeNoise(
                multiplier=(0.9, 1.1),       # Moderate increase
                p=0.4
            ),
        ], p=0.6),

        # Quality degradation
        A.OneOf([
            A.ImageCompression(
                quality_range=(30, 80),      # Moderate quality range
                p=0.4
            ),
            A.Downscale(
                scale_range=(0.4, 0.7),      # Moderate downscaling
                p=0.3
            ),
        ], p=0.5),

        # Enhanced lighting effects
        A.OneOf([
            A.RandomShadow(
                shadow_roi=(0, 0.1, 1, 1),
                num_shadows_limit=(1, 3),    # More shadows
                p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.4
            ),
        ], p=0.5),

        # Enhanced blur effects
        A.OneOf([
            A.MotionBlur(
                blur_limit=(3, 7),          # Moderate blur increase
                p=0.4
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),          # Moderate blur increase
                p=0.4
            ),
            A.Defocus(
                radius=(2, 6),              # Moderate defocus increase
                alias_blur=(0.1, 0.3),
                p=0.3
            ),
            A.ZoomBlur(
                max_factor=1.03,            # Reduced intensity
                p=0.2
            ),
        ], p=0.5),

        # Color space transformations
        A.OneOf([
            A.ToSepia(p=0.5),
            A.ToGray(p=0.5),
            A.ChannelShuffle(p=0.3),
        ], p=0.05),

        # Enhanced geometric distortions
        A.OneOf([
            A.OpticalDistortion(
                distort_limit=0.3,          # Moderate distortion
                p=0.3
            ),
            A.GridDistortion(
                num_steps=6,                # Moderate grid steps
                distort_limit=0.3,          # Moderate distortion
                p=0.3
            ),
            A.ElasticTransform(
                alpha=1.0,                  # Moderate alpha
                sigma=25,                   # Moderate sigma
                p=0.2
            ),
            A.Perspective(
                scale=(0.05, 0.1),          # Reduced perspective
                p=0.15
            ),
        ], p=0.4),

        # Additional challenging augmentations
        A.OneOf([
            A.Emboss(
                alpha=(0.2, 0.4),           # Reduced upper bound
                strength=(0.2, 0.5),        # Reduced upper bound
                p=0.3
            ),
            A.Sharpen(
                alpha=(0.2, 0.6),           # Reduced upper bound
                lightness=(0.5, 1.2),       # Reduced upper bound
                p=0.3
            ),
            A.Posterize(
                num_bits=5,                 # Increased from 4
                p=0.2
            ),
        ], p=0.3),

        # Weather and environmental effects
        A.OneOf([
            A.RandomSnow(
                snow_point_range=(0.1, 0.3),
                brightness_coeff=2.5,
                method="bleach",
                p=0.1
            ),
            A.RandomRain(
                slant_range=(-10, 10),
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=7,
                brightness_coefficient=0.7,
                rain_type="default",
                p=0.1
            ),
        ], p=0.15),
    ]

    if mode == TransformMode.SYNTHETIC.value:
        transforms = base_transform[:-1] + geometric_transforms + synthetic_transforms + [base_transform[-1]]
    else:
        transforms = base_transform[:-1] + geometric_transforms + color_transforms + noise_transforms + [base_transform[-1]]

    return A.Compose(transforms, additional_targets={'mask': 'mask'})
