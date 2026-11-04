import numpy as np

def calcium_drive(voltage, rest_mV, scale_mV) -> np.ndarray:
    """Calculate calcium drive."""
    return np.maximum(np.asarray(voltage, dtype=float) - rest_mV, 0.0) / scale_mV

def indicator_state(drive, dt_ms, tau_rise_ms, tau_decay_ms) -> np.ndarray:
    """First-order indicator with separate rise and decay time constants, stepped along the last axis."""
    drive = np.asarray(drive, dtype=float)
    out = np.zeros_like(drive)
    x = drive[..., 0].copy()
    out[..., 0] = x
    for t in range(1, drive.shape[-1]):
        d = drive[..., t]
        rate = np.where(d > x, 1.0 / tau_rise_ms, 1.0 / tau_decay_ms)
        x = x + (d - x) * (1.0 - np.exp(-dt_ms * rate))
        out[..., t] = x
    return out

def saturate(x, half_level, hill) -> np.ndarray:
    """Hill saturation of the indicator state."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return x**hill / (x**hill + half_level**hill)

def bleach(f, dt_ms, tau_bleach_s) -> np.ndarray:
    """Bleach signal."""
    t_s = np.arange(f.shape[-1]) * dt_ms / 1000.0
    return f * np.exp(-t_s / tau_bleach_s)

def to_frames(f, dt_ms, frame_rate_hz) -> np.ndarray:
    """Convert to frames."""
    n = max(1, int(round(1000.0 / frame_rate_hz / dt_ms)))
    n_frames = f.shape[-1] // n
    return np.mean(f[..., :n_frames * n].reshape(f.shape[:-1] + (n_frames, n)), axis=-1)

def fluorescence(voltage, dt_ms, params, rng) -> np.ndarray:
    """Calculate fluorescence."""
    drive = calcium_drive(voltage, params["rest_mV"], params["scale_mV"])
    state = indicator_state(drive, dt_ms, params["tau_rise_ms"], params["tau_decay_ms"])
    sat = saturate(state, params["half_level"], params["hill"])
    bleached = bleach(sat, dt_ms, params["tau_bleach_s"])
    frames = to_frames(bleached, dt_ms, params["frame_rate_hz"])
    return frames + rng.normal(0.0, params["noise_sd"], frames.shape)
