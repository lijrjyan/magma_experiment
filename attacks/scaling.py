"""Scaling attack helper."""

def report_fake_data_size(real_data_size: int, multiplier: float = 10.0) -> int:
    return int(real_data_size * multiplier)
