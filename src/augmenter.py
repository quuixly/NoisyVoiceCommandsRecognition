import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import (
    AddGaussianNoise,
    ClippingDistortion,
    Compose,
    Gain,
    HighPassFilter,
    LowPassFilter,
    PitchShift,
    Shift,
    TimeStretch,
)
from audiomentations.core.transforms_interface import BaseWaveformTransform

from src.constants import AUG_DIR, COMMANDS_TO_PROCESS, DATA_DIR
from src.visualization import plot_comparison


class Augmenter:
    """Applies single + combo audiomentations transforms to source audio,
    with an optional interactive raw-vs-augmented preview."""

    SINGLE_TRANSFORMS: dict[str, BaseWaveformTransform] = {
        "gaussian_noise": AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=1.0),
        "time_stretch": TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
        "pitch_shift": PitchShift(min_semitones=-3, max_semitones=3, p=1.0),
        "shift": Shift(min_shift=-0.2, max_shift=0.2, p=1.0),
        "gain": Gain(min_gain_db=-6, max_gain_db=6, p=1.0),
        "low_pass": LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=4000, p=1.0),
        "high_pass": HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=1000, p=1.0),
        "clipping": ClippingDistortion(
            min_percentile_threshold=0, max_percentile_threshold=5, p=1.0
        ),
    }
    COMBO_SIZES = [2, 3, 4, 5]

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        aug_dir: Path = AUG_DIR,
        commands: list[str] | None = None,
        interactive: bool = False,
        num_preview_files: int = 5,
        sources_per_command: int = 100,
    ) -> None:
        self.data_dir = data_dir
        self.aug_dir = aug_dir
        self.commands = commands if commands is not None else COMMANDS_TO_PROCESS
        self.interactive = interactive
        self.num_preview_files = num_preview_files
        self.sources_per_command = sources_per_command
        self._source_to_generated: dict[Path, list[Path]] = {}  # raw path -> its generated variants
        self.manifest_rows: list[dict] = []  # built live so raw<->augmented pairing needs no disk re-derivation

    def _build_combo_transforms(self) -> dict[str, Compose]:
        combo_transforms = {}
        keys = list(self.SINGLE_TRANSFORMS.keys())

        for size in self.COMBO_SIZES:
            for combo_keys in combinations(keys, size):
                name = "+".join(combo_keys)
                transforms = [self.SINGLE_TRANSFORMS[k] for k in combo_keys]
                combo_transforms[name] = Compose(transforms)

        return combo_transforms

    def _apply_and_save(
        self,
        samples: np.ndarray,
        sample_rate: int,
        transform,
        out_path: Path,
    ) -> None:
        """Apply one transform and save the resulting .wav file."""
        try:
            augmented = transform(
                samples=samples.astype(np.float32), sample_rate=sample_rate
            )
            sf.write(out_path, augmented, sample_rate)
        except Exception as e:
            logging.warning(f"  Transform failed for {out_path.name}: {e}")

    def augment_audio(self) -> None:
        """
        For every source file produce:
          - one .wav per single transform
          - one .wav per combination of transforms (sizes 2-5)
        Output structure: augmented/<command>/<command>_<its number>_<transform_name>.wav
        """
        self.aug_dir.mkdir(parents=True, exist_ok=True)
        self._source_to_generated.clear()
        self.manifest_rows.clear()

        all_transforms: dict[str, BaseWaveformTransform | Compose] = {}
        all_transforms.update(self.SINGLE_TRANSFORMS)
        all_transforms.update(self._build_combo_transforms())

        logging.info(f"Total transform variants to apply per file: {len(all_transforms)}")

        for command in self.commands:
            out_dir = self.aug_dir / command
            out_dir.mkdir(parents=True, exist_ok=True)

            source_files = sorted((self.data_dir / command).iterdir())[
                : self.sources_per_command
            ]
            logging.info(
                f"Augmenting {command}: {len(source_files)} files × {len(all_transforms)} transforms"
            )

            for idx, item in enumerate(source_files):
                if not item.is_file():
                    continue

                samples, sample_rate = sf.read(item)
                generated_paths: list[Path] = []
                # command is prefixed because idx resets to 0 for every command
                group_id = f"{command}_{idx:03d}"

                self.manifest_rows.append(
                    {
                        "file_path": item.as_posix(),
                        "command": command,
                        "group_id": group_id,
                        "kind": "raw",
                        "transform": "raw",
                    }
                )

                for transform_name, transform in all_transforms.items():
                    out_path = out_dir / f"{command}_{idx}_{transform_name}.wav"
                    if not out_path.exists():
                        self._apply_and_save(samples, sample_rate, transform, out_path)
                    if out_path.exists():
                        generated_paths.append(out_path)
                        self.manifest_rows.append(
                            {
                                "file_path": out_path.as_posix(),
                                "command": command,
                                "group_id": group_id,
                                "kind": "augmented",
                                "transform": transform_name,
                            }
                        )

                self._source_to_generated[item] = generated_paths

            logging.info(f"  Done: {command}")

        if self.interactive:
            self._preview()

    def _sample_preview_pairs(self) -> list[tuple[Path, Path]]:
        pairs = [
            (source, generated[0])
            for source, generated in self._source_to_generated.items()
            if generated
        ]
        return pairs[: self.num_preview_files]

    def _preview(self) -> None:
        pairs = self._sample_preview_pairs()
        if not pairs:
            logging.warning("No raw/augmented pairs available for preview")
            return

        for raw_file, augmented_file in pairs:
            plot_comparison(raw_file, augmented_file)
