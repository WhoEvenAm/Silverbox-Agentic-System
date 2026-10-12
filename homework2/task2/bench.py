"""Binary-amplitude transmission-matrix bench and noisy-feedback flip search."""

from dataclasses import dataclass
import numpy as np

from schemas import FocusSettings


@dataclass(frozen=True)
class BenchConfig:
    dmd_side: int = 32
    camera_side: int = 32
    transmission_seed: int = 17
    photons_per_intensity: float = 400.0
    read_noise_std: float = 0.005
    beam_waist_fraction: float = 0.38


class CameraBench:
    def __init__(self, config=BenchConfig(), noise_seed=0):
        if config.photons_per_intensity <= 0 or config.read_noise_std < 0 or config.beam_waist_fraction <= 0:
            raise ValueError("Use positive photon exposure and nonnegative read noise")
        self.config = config
        rng = np.random.default_rng(config.transmission_seed)
        shape = (config.camera_side**2, config.dmd_side**2)
        coordinates = (np.arange(config.dmd_side) - (config.dmd_side - 1) / 2) / config.dmd_side
        x, y = np.meshgrid(coordinates, coordinates)
        # The lecture shows a Gaussian input beam. Its phase stays fixed; the
        # DMD only gates this fixed amplitude envelope with binary macropixels.
        self.illumination = np.exp(-(x*x + y*y) / config.beam_waist_fraction**2)
        # Random complex coefficients model scattering phases and amplitudes.
        self.transmission = (((rng.normal(size=shape) + 1j * rng.normal(size=shape))
            / np.sqrt(2 * np.sum(self.illumination**2)))
            * self.illumination.ravel()).astype(np.complex64)
        self.noise_rng = np.random.default_rng(noise_seed)
        self._bases = {}
        self._last_size = self._last_pattern = self._last_field = None
        self.shots = 0

    def expand_pattern(self, pattern, size):
        pattern = np.asarray(pattern)
        if size < 1 or self.config.dmd_side % size:
            raise ValueError("Macropixel size must divide the DMD side")
        side = self.config.dmd_side // size
        if pattern.shape != (side, side) or not np.isin(pattern, (0, 1)).all():
            raise ValueError("Supply a correctly shaped binary amplitude pattern")
        # Every physical mirror inside one macropixel shares its on/off amplitude.
        return np.repeat(np.repeat(pattern, size, axis=0), size, axis=1)

    def _transmit(self, pattern, size):
        self.expand_pattern(pattern, size)
        flat = np.asarray(pattern, dtype=float).ravel()
        if size not in self._bases:
            side = self.config.dmd_side // size
            self._bases[size] = self.transmission.reshape(
                self.config.camera_side**2, side, size, side, size
            ).sum(axis=(2, 4)).reshape(self.config.camera_side**2, -1)
        basis = self._bases[size]
        if self._last_size == size:
            changed = np.flatnonzero(flat != self._last_pattern)
            # This only accelerates the forward simulation of a new pattern.
            # The optimizer never uses T to select which macropixels to flip.
            field = self._last_field + basis[:, changed] @ (flat[changed] - self._last_pattern[changed])
        else:
            field = basis @ flat
        self._last_size, self._last_pattern, self._last_field = size, flat.copy(), field
        return field

    def noiseless_image(self, pattern, size):
        field = self._transmit(pattern, size)
        # Squaring the complex magnitude converts field to camera intensity.
        return (np.abs(field)**2).reshape(self.config.camera_side, self.config.camera_side)

    def measure(self, pattern, size, frames=1):
        if frames < 1:
            raise ValueError("At least one camera exposure is required")
        intensity = self.noiseless_image(pattern, size)
        exposure = self.config.photons_per_intensity
        # Draw fresh noise on EVERY exposure, even when the pattern is unchanged.
        photons = self.noise_rng.poisson(intensity * exposure, size=(frames, *intensity.shape))
        read = self.noise_rng.normal(0, self.config.read_noise_std, photons.shape)
        self.shots += frames
        return np.maximum(photons / exposure + read, 0).mean(axis=0)


def camera_regions(settings, camera_side=32):
    y, x = np.indices((camera_side, camera_side))
    dx, dy = x - settings.focus_x, y - settings.focus_y
    half = settings.focus_size // 2
    focus = (np.abs(dx) <= half) & (np.abs(dy) <= half)
    radius = np.hypot(dx, dy)
    background = ((radius > settings.background_inner_radius)
                  & (radius <= settings.background_outer_radius))
    angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    indices = np.floor(angle * settings.background_regions / (2 * np.pi)).astype(int)
    sectors = [background & (indices == index) for index in range(settings.background_regions)]
    if not focus.any() or any(not sector.any() for sector in sectors):
        raise ValueError("Focus and every background region must contain camera pixels")
    if np.any(focus & background):
        raise ValueError("Focus and background must not overlap")
    return focus, sectors


def enhancement(image, focus, sectors):
    focus_mean = float(image[focus].mean())
    means = [float(image[sector].mean()) for sector in sectors]
    counts = [int(sector.sum()) for sector in sectors]
    # Partition the proximal background, then combine its regions by pixel count.
    background_mean = float(np.average(means, weights=counts))
    return {"enhancement": focus_mean / max(background_mean, 1e-12),
            "focus_mean": focus_mean, "background_mean": background_mean,
            "background_region_means": means, "background_region_pixels": counts}


def validate_pattern(bench, pattern, settings, focus, sectors):
    images, scores = [], []
    # Fresh shots estimate the chosen pattern's result independently of search.
    for _ in range(settings.validation_shots):
        image = bench.measure(pattern, settings.macropixel_size)
        images.append(image)
        scores.append(enhancement(image, focus, sectors)["enhancement"])
    return np.mean(images, axis=0), {"enhancement": float(np.mean(scores)),
        "enhancement_std": float(np.std(scores, ddof=1)),
        "enhancement_sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
        "shot_enhancements": scores}


def search_focus(settings: FocusSettings, config=BenchConfig(), progress=None):
    rng = np.random.default_rng(settings.seed)
    bench = CameraBench(config, noise_seed=settings.seed + 100000)
    side = config.dmd_side // settings.macropixel_size
    pattern = rng.integers(0, 2, size=(side, side), dtype=np.uint8)
    if not pattern.any():
        pattern[0, 0] = 1
    focus, sectors = camera_regions(settings, config.camera_side)
    initial_pattern = pattern.copy()
    initial_image, initial = validate_pattern(bench, pattern, settings, focus, sectors)
    history = []
    for iteration in range(1, settings.iterations + 1):
        proposal = pattern.copy()
        changed = rng.choice(pattern.size, settings.flip_count, replace=False)
        proposal.ravel()[changed] ^= 1
        if not proposal.any():
            continue  # A dark DMD cannot create a meaningful optical focus.
        # Remeasure both patterns: do not compare a noisy proposal to a stale score.
        baseline_image = bench.measure(pattern, settings.macropixel_size, settings.frames_per_measurement)
        candidate_image = bench.measure(proposal, settings.macropixel_size, settings.frames_per_measurement)
        baseline = enhancement(baseline_image, focus, sectors)["enhancement"]
        candidate = enhancement(candidate_image, focus, sectors)["enhancement"]
        accepted = candidate > baseline
        if accepted:
            pattern = proposal
        history.append({"iteration": iteration, "baseline": baseline,
                        "candidate": candidate, "accepted": accepted,
                        "retained_measurement": candidate if accepted else baseline})
        if progress and (iteration == 1 or iteration % max(1, settings.iterations // 10) == 0):
            progress(iteration, history[-1], candidate_image if accepted else baseline_image)
    final_image, final = validate_pattern(bench, pattern, settings, focus, sectors)
    return {"settings": settings.model_dump(), "initial": initial, "final": final,
            "initial_image": initial_image, "final_image": final_image,
            "initial_pattern": initial_pattern, "final_pattern": pattern,
            "focus_mask": focus, "background_masks": sectors, "history": history,
            "camera_shots": bench.shots, "accepted_flips": sum(row["accepted"] for row in history),
            "final_active_fraction": float(pattern.mean())}
