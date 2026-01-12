from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from device_anomaly.data_access import mc_timeseries_loader


def test_stream_mc_timeseries_stops_on_non_advancing_watermark(monkeypatch):
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame([{"DeviceId": 1, "ServerDateTime": start_time}])

    def fake_load(*args, **kwargs):
        return df, start_time

    monkeypatch.setattr(mc_timeseries_loader, "load_mc_timeseries_incremental", fake_load)

    batches = list(
        mc_timeseries_loader.stream_mc_timeseries(
            table_name="DeviceStatInt",
            start_time=start_time,
            engine=object(),
        )
    )

    assert len(batches) == 1
