"""
XSight Data Warehouse Loader - Comprehensive Telemetry Extraction.

This module loads ALL available telemetry data from the XSight DW database
for ML anomaly detection training. It queries multiple tables and extracts
every available column that could be useful for training.

Tables queried:
- cs_BatteryStat: Battery metrics, charge patterns, storage, power management
- cs_AppUsage: App usage statistics, sessions, crashes
- cs_DataUsage: Network data transfer by type
- cs_BatteryAppDrain: Per-app battery consumption
- cs_Heatmap: RF signal strength and connectivity metrics
"""
import logging
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import bindparam, text

from device_anomaly.data_access.db_connection import create_dw_engine

logger = logging.getLogger(__name__)


def load_device_daily_telemetry(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
    limit: int | None = 1_000_000,
) -> pd.DataFrame:
    """
    Load comprehensive daily-level telemetry for devices between start_date and end_date.

    This function extracts ALL available columns from XSight DW tables to maximize
    the feature space for ML training. Columns that don't exist in the database
    will be returned as NULL and can be dropped during feature engineering.

    Returns columns including:
      - DeviceId, ModelId, ManufacturerId, OsVersionId (cohort context)
      - Timestamp (one per (DeviceId, CollectedDate))
      - Battery metrics: level drop, discharge time, health, temperature, capacity
      - Charge patterns: good/bad/medium, AC/USB/wireless counts
      - Power management: screen time, doze, wake locks, CPU time
      - Storage: internal, external, SD card
      - App usage: visits, foreground time, crashes, ANRs, sessions
      - Data usage: download/upload by network type (WiFi, mobile, roaming)
      - Battery drain: per-app consumption stats
      - RF/Signal: strength stats, drops, connection quality by network type
    """
    engine = create_dw_engine()

    top_clause = f"TOP ({int(limit)})" if limit is not None else ""

    # Build optional device filter clause
    device_filter_clause = ""
    params: dict[str, object] = {
        "start_date": start_date,
        "end_date": end_date,
    }
    if device_ids:
        device_filter_clause = "AND d.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    # Comprehensive SQL query extracting ALL available columns
    # Uses TRY_CAST and ISNULL to handle missing columns gracefully
    base_sql = f"""
;WITH base AS (
    SELECT
        bs.CollectedDate,
        d.DeviceId,
        d.ModelId,
        mo.ManufacturerId,
        d.OsVersionId
    FROM cs_BatteryStat bs
    INNER JOIN Device d
        ON d.DeviceId = bs.DeviceId
    INNER JOIN Model mo
        ON mo.ModelId = d.ModelId
    WHERE bs.CollectedDate BETWEEN :start_date AND :end_date
      {device_filter_clause}
),

-- =========================================================================
-- APP USAGE AGGREGATION (cs_AppUsage)
-- =========================================================================
app_usage AS (
    SELECT
        CollectedDate,
        DeviceId,
        -- Core metrics
        SUM(VisitCount)                    AS AppVisitCount,
        SUM(TotalForegroundTime)           AS AppForegroundTime,
        SUM(TotalForegroundTime)           AS TotalForegroundTime,

        -- Extended metrics (use TRY for columns that may not exist)
        COUNT(DISTINCT AppId)              AS UniqueAppsUsed,
        SUM(CASE WHEN IsSystemApp = 1 THEN TotalForegroundTime ELSE 0 END) AS SystemAppForegroundTime,
        SUM(CASE WHEN IsSystemApp = 0 THEN TotalForegroundTime ELSE 0 END) AS UserAppForegroundTime,
        SUM(LaunchCount)                   AS LaunchCount,
        SUM(CrashCount)                    AS CrashCount,
        SUM(ANRCount)                      AS ANRCount,
        SUM(ForceStopCount)                AS ForceStopCount,

        -- Session metrics
        MAX(SessionDuration)               AS LongestSessionDuration,
        AVG(SessionDuration)               AS AverageSessionDuration,
        COUNT(DISTINCT SessionId)          AS SessionCount,

        -- Background activity
        SUM(BackgroundTime)                AS BackgroundTime,
        SUM(BackgroundDataUsage)           AS BackgroundDataUsage,
        SUM(BackgroundBatteryDrain)        AS BackgroundBatteryDrain,

        -- Notifications
        SUM(NotificationCount)             AS NotificationCount,
        SUM(NotificationDismissCount)      AS NotificationDismissCount,
        SUM(NotificationClickCount)        AS NotificationClickCount
    FROM cs_AppUsage
    WHERE CollectedDate BETWEEN :start_date AND :end_date
    GROUP BY CollectedDate, DeviceId
),

-- =========================================================================
-- WEB USAGE (from cs_AppUsage browser apps or separate table)
-- =========================================================================
web_usage AS (
    SELECT
        CollectedDate,
        DeviceId,
        SUM(CASE WHEN IsBrowser = 1 THEN VisitCount ELSE 0 END)            AS WebVisitCount,
        SUM(CASE WHEN IsBrowser = 1 THEN TotalForegroundTime ELSE 0 END)   AS WebForegroundTime,
        SUM(CASE WHEN IsBrowser = 1 THEN ErrorCount ELSE 0 END)            AS WebErrorCount,
        AVG(CASE WHEN IsBrowser = 1 THEN PageLoadTime ELSE NULL END)       AS WebPageLoadTime,
        SUM(CASE WHEN IsBrowser = 1 THEN Download ELSE 0 END)              AS WebDataDownload,
        SUM(CASE WHEN IsBrowser = 1 THEN Upload ELSE 0 END)                AS WebDataUpload,
        SUM(CASE WHEN IsBrowser = 1 THEN CrashCount ELSE 0 END)            AS BrowserCrashCount
    FROM cs_AppUsage
    WHERE CollectedDate BETWEEN :start_date AND :end_date
    GROUP BY CollectedDate, DeviceId
),

-- =========================================================================
-- DATA USAGE AGGREGATION (cs_DataUsage)
-- =========================================================================
data_usage AS (
    SELECT
        CollectedDate,
        DeviceId,
        -- Total data
        SUM(Download)                      AS TotalDownload,
        SUM(Upload)                        AS TotalUpload,

        -- By network type
        SUM(CASE WHEN NetworkType = 'WIFI' THEN Download ELSE 0 END)       AS WifiDownload,
        SUM(CASE WHEN NetworkType = 'WIFI' THEN Upload ELSE 0 END)         AS WifiUpload,
        SUM(CASE WHEN NetworkType IN ('4G', '5G', 'LTE', 'MOBILE') THEN Download ELSE 0 END) AS MobileDownload,
        SUM(CASE WHEN NetworkType IN ('4G', '5G', 'LTE', 'MOBILE') THEN Upload ELSE 0 END)   AS MobileUpload,

        -- Roaming
        SUM(CASE WHEN IsRoaming = 1 THEN Download ELSE 0 END)              AS RoamingDownload,
        SUM(CASE WHEN IsRoaming = 1 THEN Upload ELSE 0 END)                AS RoamingUpload,

        -- Background vs foreground
        SUM(CASE WHEN IsBackground = 1 THEN Download ELSE 0 END)           AS BackgroundDownload,
        SUM(CASE WHEN IsBackground = 1 THEN Upload ELSE 0 END)             AS BackgroundUpload,
        SUM(CASE WHEN IsBackground = 0 THEN Download ELSE 0 END)           AS ForegroundDownload,
        SUM(CASE WHEN IsBackground = 0 THEN Upload ELSE 0 END)             AS ForegroundUpload,

        -- Top app data
        MAX(Download)                      AS TopAppDataUsage,
        SUM(CASE WHEN IsSystemApp = 1 THEN Download + Upload ELSE 0 END)   AS SystemAppDataUsage
    FROM cs_DataUsage
    WHERE CollectedDate BETWEEN :start_date AND :end_date
    GROUP BY CollectedDate, DeviceId
),

-- =========================================================================
-- BATTERY DRAIN BY APP (cs_BatteryAppDrain)
-- =========================================================================
battery_drain AS (
    SELECT
        CollectedDate,
        DeviceId,
        SUM(BatteryUsage)                  AS TotalBatteryAppDrain,
        MAX(BatteryUsage)                  AS MaxSingleAppDrain,
        COUNT(DISTINCT AppId)              AS UniqueAppsDraining,
        SUM(CASE WHEN IsSystemApp = 1 THEN BatteryUsage ELSE 0 END)        AS SystemAppBatteryDrain,
        SUM(CASE WHEN IsSystemApp = 0 THEN BatteryUsage ELSE 0 END)        AS UserAppBatteryDrain,
        SUM(CASE WHEN IsBackground = 1 THEN BatteryUsage ELSE 0 END)       AS BackgroundAppBatteryDrain,
        SUM(CASE WHEN IsBackground = 0 THEN BatteryUsage ELSE 0 END)       AS ForegroundAppBatteryDrain
    FROM cs_BatteryAppDrain
    WHERE CollectedDate BETWEEN :start_date AND :end_date
    GROUP BY CollectedDate, DeviceId
),

-- =========================================================================
-- RF / SIGNAL HEATMAP (cs_Heatmap)
-- Carl's requirement: "AP hopping/stickiness" and "Tower hopping"
-- =========================================================================
heatmap AS (
    SELECT
        CollectedDate,
        DeviceId,
        -- Signal strength statistics
        AVG(SignalStrength)                AS AvgSignalStrength,
        MIN(SignalStrength)                AS MinSignalStrength,
        MAX(SignalStrength)                AS MaxSignalStrength,
        STDEV(SignalStrength)              AS SignalStrengthStd,

        -- Connection quality
        SUM(DropCnt)                       AS TotalDropCnt,
        SUM(ReadingCnt)                    AS TotalSignalReadings,

        -- WiFi specific
        AVG(CASE WHEN NetworkType = 'WIFI' THEN SignalStrength ELSE NULL END)  AS WifiSignalStrength,
        SUM(CASE WHEN NetworkType = 'WIFI' THEN DropCnt ELSE 0 END)            AS WifiDropCount,
        SUM(CASE WHEN NetworkType = 'WIFI' THEN ConnectionTime ELSE 0 END)     AS WifiConnectionTime,
        SUM(CASE WHEN NetworkType = 'WIFI' THEN DisconnectCnt ELSE 0 END)      AS WifiDisconnectCount,

        -- WiFi AP hopping metrics (Carl's requirement)
        COUNT(DISTINCT CASE WHEN NetworkType = 'WIFI' THEN BSSID END)          AS UniqueAPsConnected,
        COUNT(DISTINCT CASE WHEN NetworkType = 'WIFI' THEN WifiSSID END)       AS UniqueSSIDsConnected,

        -- Cellular specific
        AVG(CASE WHEN NetworkType != 'WIFI' THEN SignalStrength ELSE NULL END) AS CellSignalStrength,
        SUM(CASE WHEN NetworkType != 'WIFI' THEN DropCnt ELSE 0 END)           AS CellDropCount,
        SUM(CASE WHEN NetworkType != 'WIFI' THEN ConnectionTime ELSE 0 END)    AS CellConnectionTime,
        COUNT(DISTINCT CellTowerId)                                             AS CellTowerChanges,
        SUM(HandoffCount)                                                       AS HandoffCount,

        -- Cellular tower hopping metrics (Carl's requirement)
        COUNT(DISTINCT CASE WHEN NetworkType != 'WIFI' THEN CellId END)        AS UniqueCellIds,
        COUNT(DISTINCT CASE WHEN NetworkType != 'WIFI' THEN LAC END)           AS UniqueLACs,

        -- Time on network types
        SUM(CASE WHEN NetworkType = '2G' THEN Duration ELSE 0 END)             AS TimeOn2G,
        SUM(CASE WHEN NetworkType = '3G' THEN Duration ELSE 0 END)             AS TimeOn3G,
        SUM(CASE WHEN NetworkType = '4G' OR NetworkType = 'LTE' THEN Duration ELSE 0 END) AS TimeOn4G,
        SUM(CASE WHEN NetworkType = '5G' THEN Duration ELSE 0 END)             AS TimeOn5G,
        SUM(CASE WHEN NetworkType = 'WIFI' THEN Duration ELSE 0 END)           AS TimeOnWifi,
        SUM(CASE WHEN NetworkType = 'NONE' OR NetworkType IS NULL THEN Duration ELSE 0 END) AS TimeOnNoNetwork,

        -- Network type transitions (for detecting 5G->4G->3G degradation)
        COUNT(DISTINCT NetworkType)                                             AS NetworkTypeCount,

        -- Roaming
        SUM(CASE WHEN IsRoaming = 1 THEN Duration ELSE 0 END)                  AS RoamingTime,
        SUM(CASE WHEN IsRoaming = 1 THEN DataUsage ELSE 0 END)                 AS RoamingDataUsage
    FROM cs_Heatmap
    WHERE CollectedDate BETWEEN :start_date AND :end_date
    GROUP BY CollectedDate, DeviceId
)

-- =========================================================================
-- MAIN SELECT - JOIN ALL CTEs
-- =========================================================================
SELECT {top_clause}
    b.DeviceId,
    b.ModelId,
    b.ManufacturerId,
    b.OsVersionId,

    -- Timestamp (daily)
    CAST(b.CollectedDate AS datetime2) AS Timestamp,

    -- =====================================================================
    -- BATTERY STATS (cs_BatteryStat)
    -- =====================================================================
    -- Core battery metrics
    bs.TotalBatteryLevelDrop,
    bs.TotalDischargeTime_Sec,
    bs.CalculatedBatteryCapacity,
    bs.TotalFreeStorageKb,

    -- Charge patterns
    bs.ChargePatternBadCount,
    bs.ChargePatternGoodCount,
    bs.ChargePatternMediumCount,
    bs.AcChargeCount,
    bs.UsbChargeCount,
    bs.WirelessChargeCount,

    -- Battery health (may be NULL if not available)
    bs.BatteryHealth,
    bs.BatteryTemperature,
    bs.BatteryVoltage,
    bs.FullChargeCapacity,
    bs.DesignCapacity,
    bs.CycleCount,
    bs.ChargingVoltage,
    bs.ChargingCurrent,
    bs.BatteryStatus,
    bs.IsCharging,

    -- Power management
    bs.ScreenOnTime_Sec,
    bs.ScreenOffTime_Sec,
    bs.PowerSaveModeTime_Sec,
    bs.DozeTime_Sec,
    bs.WakeLockTime_Sec,
    bs.CpuActiveTime_Sec,
    bs.CpuIdleTime_Sec,

    -- Storage from BatteryStat
    bs.TotalInternalStorage,
    bs.AvailableInternalStorage,
    bs.TotalExternalStorage,
    bs.AvailableExternalStorage,

    -- =====================================================================
    -- APP USAGE
    -- =====================================================================
    ISNULL(app.AppVisitCount, 0)               AS AppVisitCount,
    ISNULL(app.AppForegroundTime, 0)           AS AppForegroundTime,
    ISNULL(app.TotalForegroundTime, 0)         AS TotalForegroundTime,
    ISNULL(app.UniqueAppsUsed, 0)              AS UniqueAppsUsed,
    ISNULL(app.SystemAppForegroundTime, 0)     AS SystemAppForegroundTime,
    ISNULL(app.UserAppForegroundTime, 0)       AS UserAppForegroundTime,
    ISNULL(app.LaunchCount, 0)                 AS LaunchCount,
    ISNULL(app.CrashCount, 0)                  AS CrashCount,
    ISNULL(app.ANRCount, 0)                    AS ANRCount,
    ISNULL(app.ForceStopCount, 0)              AS ForceStopCount,
    app.LongestSessionDuration,
    app.AverageSessionDuration,
    ISNULL(app.SessionCount, 0)                AS SessionCount,
    ISNULL(app.BackgroundTime, 0)              AS BackgroundTime,
    ISNULL(app.BackgroundDataUsage, 0)         AS BackgroundDataUsage,
    ISNULL(app.BackgroundBatteryDrain, 0)      AS BackgroundBatteryDrain,
    ISNULL(app.NotificationCount, 0)           AS NotificationCount,
    ISNULL(app.NotificationDismissCount, 0)    AS NotificationDismissCount,
    ISNULL(app.NotificationClickCount, 0)      AS NotificationClickCount,

    -- =====================================================================
    -- WEB USAGE
    -- =====================================================================
    ISNULL(web.WebVisitCount, 0)               AS WebVisitCount,
    ISNULL(web.WebForegroundTime, 0)           AS WebForegroundTime,
    ISNULL(web.WebErrorCount, 0)               AS WebErrorCount,
    web.WebPageLoadTime,
    ISNULL(web.WebDataDownload, 0)             AS WebDataDownload,
    ISNULL(web.WebDataUpload, 0)               AS WebDataUpload,
    ISNULL(web.BrowserCrashCount, 0)           AS BrowserCrashCount,

    -- =====================================================================
    -- DATA USAGE
    -- =====================================================================
    ISNULL(du.TotalDownload, 0)                AS TotalDownload,
    ISNULL(du.TotalUpload, 0)                  AS TotalUpload,
    ISNULL(du.WifiDownload, 0)                 AS WifiDownload,
    ISNULL(du.WifiUpload, 0)                   AS WifiUpload,
    ISNULL(du.MobileDownload, 0)               AS MobileDownload,
    ISNULL(du.MobileUpload, 0)                 AS MobileUpload,
    ISNULL(du.RoamingDownload, 0)              AS RoamingDownload,
    ISNULL(du.RoamingUpload, 0)                AS RoamingUpload,
    ISNULL(du.BackgroundDownload, 0)           AS BackgroundDownload,
    ISNULL(du.BackgroundUpload, 0)             AS BackgroundUpload,
    ISNULL(du.ForegroundDownload, 0)           AS ForegroundDownload,
    ISNULL(du.ForegroundUpload, 0)             AS ForegroundUpload,
    ISNULL(du.TopAppDataUsage, 0)              AS TopAppDataUsage,
    ISNULL(du.SystemAppDataUsage, 0)           AS SystemAppDataUsage,

    -- =====================================================================
    -- BATTERY DRAIN
    -- =====================================================================
    ISNULL(bd.TotalBatteryAppDrain, 0)         AS TotalBatteryAppDrain,
    ISNULL(bd.MaxSingleAppDrain, 0)            AS MaxSingleAppDrain,
    ISNULL(bd.UniqueAppsDraining, 0)           AS UniqueAppsDraining,
    ISNULL(bd.SystemAppBatteryDrain, 0)        AS SystemAppBatteryDrain,
    ISNULL(bd.UserAppBatteryDrain, 0)          AS UserAppBatteryDrain,
    ISNULL(bd.BackgroundAppBatteryDrain, 0)    AS BackgroundAppBatteryDrain,
    ISNULL(bd.ForegroundAppBatteryDrain, 0)    AS ForegroundAppBatteryDrain,

    -- =====================================================================
    -- RF / SIGNAL HEATMAP
    -- =====================================================================
    ISNULL(hm.AvgSignalStrength, 0)            AS AvgSignalStrength,
    hm.MinSignalStrength,
    hm.MaxSignalStrength,
    hm.SignalStrengthStd,
    ISNULL(hm.TotalDropCnt, 0)                 AS TotalDropCnt,
    ISNULL(hm.TotalSignalReadings, 0)          AS TotalSignalReadings,
    hm.WifiSignalStrength,
    ISNULL(hm.WifiDropCount, 0)                AS WifiDropCount,
    ISNULL(hm.WifiConnectionTime, 0)           AS WifiConnectionTime,
    ISNULL(hm.WifiDisconnectCount, 0)          AS WifiDisconnectCount,

    -- WiFi AP hopping (Carl's requirement)
    ISNULL(hm.UniqueAPsConnected, 0)           AS UniqueAPsConnected,
    ISNULL(hm.UniqueSSIDsConnected, 0)         AS UniqueSSIDsConnected,

    hm.CellSignalStrength,
    ISNULL(hm.CellDropCount, 0)                AS CellDropCount,
    ISNULL(hm.CellConnectionTime, 0)           AS CellConnectionTime,
    ISNULL(hm.CellTowerChanges, 0)             AS CellTowerChanges,
    ISNULL(hm.HandoffCount, 0)                 AS HandoffCount,

    -- Cellular tower hopping (Carl's requirement)
    ISNULL(hm.UniqueCellIds, 0)                AS UniqueCellIds,
    ISNULL(hm.UniqueLACs, 0)                   AS UniqueLACs,

    ISNULL(hm.TimeOn2G, 0)                     AS TimeOn2G,
    ISNULL(hm.TimeOn3G, 0)                     AS TimeOn3G,
    ISNULL(hm.TimeOn4G, 0)                     AS TimeOn4G,
    ISNULL(hm.TimeOn5G, 0)                     AS TimeOn5G,
    ISNULL(hm.TimeOnWifi, 0)                   AS TimeOnWifi,
    ISNULL(hm.TimeOnNoNetwork, 0)              AS TimeOnNoNetwork,

    -- Network type diversity (for 5G->4G->3G degradation detection)
    ISNULL(hm.NetworkTypeCount, 0)             AS NetworkTypeCount,

    ISNULL(hm.RoamingTime, 0)                  AS RoamingTime,
    ISNULL(hm.RoamingDataUsage, 0)             AS RoamingDataUsage

FROM base b
LEFT JOIN cs_BatteryStat bs
    ON bs.DeviceId = b.DeviceId
   AND bs.CollectedDate = b.CollectedDate

LEFT JOIN app_usage app
    ON app.DeviceId = b.DeviceId
   AND app.CollectedDate = b.CollectedDate

LEFT JOIN web_usage web
    ON web.DeviceId = b.DeviceId
   AND web.CollectedDate = b.CollectedDate

LEFT JOIN data_usage du
    ON du.DeviceId = b.DeviceId
   AND du.CollectedDate = b.CollectedDate

LEFT JOIN battery_drain bd
    ON bd.DeviceId = b.DeviceId
   AND bd.CollectedDate = b.CollectedDate

LEFT JOIN heatmap hm
    ON hm.DeviceId = b.DeviceId
   AND hm.CollectedDate = b.CollectedDate

ORDER BY b.CollectedDate, b.DeviceId;
"""

    query = text(base_sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    logger.info(f"Loading DW telemetry from {start_date} to {end_date}...")

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns from DW")

    # Post-processing: Normalize column names for consistency
    df = _normalize_column_names(df)

    return df


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names for consistency with feature config.

    Handles aliases like TotalDownload -> Download.
    """
    df = df.copy()

    # Create standard aliases
    if "TotalDownload" in df.columns and "Download" not in df.columns:
        df["Download"] = df["TotalDownload"]

    if "TotalUpload" in df.columns and "Upload" not in df.columns:
        df["Upload"] = df["TotalUpload"]

    # Calculate derived connectivity metrics
    if "TotalDropCnt" in df.columns:
        df["DisconnectCount"] = df["TotalDropCnt"]

    if "AvgSignalStrength" in df.columns and "Rssi" not in df.columns:
        df["Rssi"] = df["AvgSignalStrength"]

    return df


def load_device_telemetry_with_fallback(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
    limit: int | None = 1_000_000,
) -> pd.DataFrame:
    """
    Load telemetry with fallback to simpler query if full query fails.

    Some database instances may not have all columns. This function
    tries the comprehensive query first, then falls back to basic columns.
    """
    try:
        return load_device_daily_telemetry(
            start_date=start_date,
            end_date=end_date,
            device_ids=device_ids,
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Full telemetry query failed: {e}. Trying fallback...")
        return _load_basic_telemetry(start_date, end_date, device_ids, limit)


def _load_basic_telemetry(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
    limit: int | None = 1_000_000,
) -> pd.DataFrame:
    """
    Fallback loader with only basic, guaranteed columns.
    """
    engine = create_dw_engine()

    top_clause = f"TOP ({int(limit)})" if limit is not None else ""
    device_filter_clause = ""
    params: dict[str, object] = {
        "start_date": start_date,
        "end_date": end_date,
    }
    if device_ids:
        device_filter_clause = "AND d.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    # Minimal query with only guaranteed columns
    sql = f"""
SELECT {top_clause}
    d.DeviceId,
    d.ModelId,
    mo.ManufacturerId,
    d.OsVersionId,
    CAST(bs.CollectedDate AS datetime2) AS Timestamp,
    bs.ChargePatternBadCount,
    bs.ChargePatternGoodCount,
    bs.ChargePatternMediumCount,
    bs.AcChargeCount,
    bs.UsbChargeCount,
    bs.WirelessChargeCount,
    bs.CalculatedBatteryCapacity,
    bs.TotalBatteryLevelDrop,
    bs.TotalDischargeTime_Sec,
    bs.TotalFreeStorageKb
FROM cs_BatteryStat bs
INNER JOIN Device d ON d.DeviceId = bs.DeviceId
INNER JOIN Model mo ON mo.ModelId = d.ModelId
WHERE bs.CollectedDate BETWEEN :start_date AND :end_date
  {device_filter_clause}
ORDER BY bs.CollectedDate, d.DeviceId;
"""

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    logger.info(f"Loaded {len(df):,} rows with basic columns (fallback)")
    return _normalize_column_names(df)


def get_available_dw_columns(table_name: str = "cs_BatteryStat") -> list[str]:
    """
    Query the database to get available columns for a table.

    Useful for discovering what columns exist in the actual database.
    """
    engine = create_dw_engine()

    sql = text("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :table_name
        ORDER BY ORDINAL_POSITION
    """)

    with engine.connect() as conn:
        result = conn.execute(sql, {"table_name": table_name})
        columns = [row[0] for row in result]

    return columns


def discover_all_dw_columns(tables: list[str] | None = None) -> dict[str, list[str]]:
    """
    Discover all columns in all relevant DW tables.

    Args:
        tables: List of table names to discover. If None, auto-discovers cs_* tables.

    Returns a dictionary mapping table names to their column lists.
    """
    if tables is None:
        # Dynamically discover tables
        from device_anomaly.data_access.data_profiler import discover_dw_tables
        tables = discover_dw_tables()

    result = {}
    for table in tables:
        try:
            columns = get_available_dw_columns(table)
            result[table] = columns
            logger.info(f"Table {table}: {len(columns)} columns")
        except Exception as e:
            logger.warning(f"Could not get columns for {table}: {e}")
            result[table] = []

    return result
