import numpy as np
import pandas as pd


def generate_synthetic_device_telemetry(
    n_devices: int = 10,
    n_days: int = 7,
    freq: str = "1h",
    anomaly_rate: float = 0.03,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic device telemetry data that loosely matches the DW data.

    Columns:
        - DeviceId
        - Timestamp
        - TotalBatteryLevelDrop
        - TotalFreeStorageKb
        - Download
        - Upload
        - OfflineTime
        - DisconnectCount
        - WiFiSignalStrength
        - ConnectionTime
        - is_injected_anomaly (bool)
    """
    rng = np.random.default_rng(random_seed)

    timestamps = pd.date_range("2025-01-01", periods=n_days, freq="D")
    all_rows = []

    for device_id in range(1, n_devices + 1):
        # Base characteristics per device
        base_battery_drop = rng.uniform(5, 20)             # daily discharge index
        base_storage_free = rng.integers(2_000_000, 8_000_000)  # KB
        base_wifi_signal = rng.integers(40, 80)            # pseudo dBm
        base_download_hour = rng.integers(20_000, 200_000) # bytes / KB units
        base_upload_hour = rng.integers(5_000, 50_000)

        for date in timestamps:
            # day-level variations
            day_battery_drop = base_battery_drop + rng.normal(0, 3)
            day_storage = base_storage_free + rng.normal(0, 50_000)

            day_hours = pd.date_range(date, periods=int(24 * (pd.Timedelta("1D") / pd.Timedelta(freq))), freq=freq)
            for ts in day_hours:
                hour = ts.hour

                # simple diurnal pattern: more usage during "business" hours
                if 9 <= hour <= 18:
                    usage_factor = 1.5
                elif 7 <= hour < 9 or 18 < hour <= 22:
                    usage_factor = 0.8
                else:
                    usage_factor = 0.3

                download = max(
                    0,
                    usage_factor * base_download_hour + rng.normal(0, 15_000),
                )
                upload = max(
                    0,
                    usage_factor * base_upload_hour + rng.normal(0, 5_000),
                )

                # offline & disconnect counts: small Poisson around usage_factor
                offline_time = max(0, rng.poisson(2 if usage_factor > 1 else 5))
                disconnects = max(0, rng.poisson(0.5 if usage_factor > 1 else 1.5))

                wifi_signal = int(
                    np.clip(base_wifi_signal + rng.normal(0, 5), 0, 100)
                )
                connection_time = max(
                    0,
                    int(60 * usage_factor + rng.normal(0, 10)),
                )

                all_rows.append(
                    {
                        "DeviceId": device_id,
                        "Timestamp": ts,
                        "TotalBatteryLevelDrop": max(0, int(day_battery_drop)),
                        "TotalFreeStorageKb": max(0, int(day_storage)),
                        "Download": int(download),
                        "Upload": int(upload),
                        "OfflineTime": int(offline_time),
                        "DisconnectCount": int(disconnects),
                        "WiFiSignalStrength": wifi_signal,
                        "ConnectionTime": connection_time,
                        "is_injected_anomaly": False,
                    }
                )

        # Inject anomalies for this device across its rows
        device_start = (device_id - 1) * len(timestamps) * len(
            pd.date_range(timestamps[0], periods=int(24 * (pd.Timedelta("1D") / pd.Timedelta(freq))), freq=freq)
        )
        device_end = device_start + len(timestamps) * len(
            pd.date_range(timestamps[0], periods=int(24 * (pd.Timedelta("1D") / pd.Timedelta(freq))), freq=freq)
        )

        n_rows_device = device_end - device_start
        n_anomalies = max(1, int(anomaly_rate * n_rows_device))

        anomaly_indices = rng.choice(
            np.arange(device_start, device_end),
            size=n_anomalies,
            replace=False,
        )

        for idx in anomaly_indices:
            row = all_rows[idx]
            anomaly_type = rng.choice(
                ["battery_spike", "storage_low", "traffic_burst", "massive_offline", "wifi_drop"]
            )

            if anomaly_type == "battery_spike":
                row["TotalBatteryLevelDrop"] += rng.integers(30, 80)
            elif anomaly_type == "storage_low":
                row["TotalFreeStorageKb"] = max(
                    0,
                    row["TotalFreeStorageKb"] - rng.integers(500_000, 2_000_000),
                )
            elif anomaly_type == "traffic_burst":
                row["Download"] *= rng.integers(5, 15)
                row["Upload"] *= rng.integers(5, 15)
            elif anomaly_type == "massive_offline":
                row["OfflineTime"] += rng.integers(30, 120)
                row["DisconnectCount"] += rng.integers(5, 20)
            elif anomaly_type == "wifi_drop":
                row["WiFiSignalStrength"] = max(
                    0,
                    row["WiFiSignalStrength"] - rng.integers(30, 70),
                )

            row["is_injected_anomaly"] = True

    df = pd.DataFrame(all_rows)
    df.sort_values(["DeviceId", "Timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
