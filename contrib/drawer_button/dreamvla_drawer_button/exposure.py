"""Keep raw-frame epochs distinct from passes over valid action-chunk starts."""

import math
from decimal import Decimal, ROUND_CEILING


def budget(episode_lengths, epochs=5.7, batch_size=16, horizon=40):
    if (
        not episode_lengths
        or not math.isfinite(epochs)
        or epochs <= 0
        or type(batch_size) is not int
        or batch_size < 1
        or type(horizon) is not int
        or horizon < 1
        or any(type(n) is not int or n < horizon for n in episode_lengths)
    ):
        raise ValueError("Invalid frame exposure configuration.")
    frames = sum(episode_lengths)
    windows = sum(n - horizon + 1 for n in episode_lengths)
    samples = int(
        (Decimal(frames) * Decimal(str(epochs))).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    return dict(
        frames=frames,
        windows=windows,
        samples=samples,
        updates=math.ceil(samples / batch_size),
        batch_size=batch_size,
        final_batch_size=(samples - 1) % batch_size + 1,
        raw_frame_epochs=epochs,
        realized_raw_frame_epochs=samples / frames,
        average_window_passes=samples / windows,
        horizon=horizon,
        end_padding=False,
    )


def next_batch(order, rng, samples_seen, total_samples, batch_size=16):
    if not len(order) or not 0 <= samples_seen < total_samples or batch_size < 1:
        raise ValueError("Invalid sample cursor or batch size.")
    stop = min(samples_seen + batch_size, total_samples)
    result = []
    while samples_seen < stop:
        cursor = samples_seen % len(order)
        if cursor == 0:
            rng.shuffle(order)
        count = min(stop - samples_seen, len(order) - cursor)
        result.extend(int(i) for i in order[cursor : cursor + count])
        samples_seen += count
    return result
