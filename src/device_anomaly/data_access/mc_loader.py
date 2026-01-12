"""
MobiControl Device Inventory Loader - Comprehensive Device Metadata Extraction.

This module loads ALL available device metadata from the MobiControl database
for ML anomaly detection training. It extracts device identity, hardware specs,
security status, network info, and platform-specific features.

Tables queried:
- DevInfo: Core device information
- iOSDevice: iOS-specific features
- WindowsDevice: Windows-specific features
- MacDevice: Mac-specific features
- MacDeviceSecurity: Mac security features
- LinuxDevice: Linux-specific features
- ZebraAndroidDevice: Zebra Android-specific features
- LabelDevice/LabelType: Custom device labels
"""
import logging
from typing import Iterable

import pandas as pd
from sqlalchemy import bindparam, text

from device_anomaly.data_access.db_connection import create_mc_engine

logger = logging.getLogger(__name__)


def load_mc_device_inventory_snapshot(
    start_dt: str,
    end_dt: str,
    device_ids: Iterable[int] | None = None,
    include_labels: bool = True,
    limit: int | None = 1_000_000,
) -> pd.DataFrame:
    """
    Load comprehensive MC device inventory snapshots for devices whose LastCheckInTime
    falls between start_dt and end_dt.

    This function extracts ALL available columns from MobiControl tables to maximize
    the feature space for ML training, including:
      - Core DevInfo fields (identity, OS/HW, storage/RAM, battery, security, network)
      - Platform enrichments (iOS/Windows/Mac/Linux/Zebra)
      - Security compliance indicators
      - Network and connectivity status
      - Optional labels in long form (DeviceId, LabelName, LabelValue)
    """
    engine = create_mc_engine()
    top_clause = f"TOP ({int(limit)})" if limit is not None else ""

    device_filter_clause = ""
    params: dict[str, object] = {"start_dt": start_dt, "end_dt": end_dt}
    if device_ids:
        device_filter_clause = "AND d.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    labels_cte = ""
    labels_join = ""
    labels_select = ""

    if include_labels:
        labels_cte = """
, labels AS (
    SELECT
        ld.DeviceId,
        lt.Name AS LabelName,
        CONVERT(nvarchar(4000), ld.Value) AS LabelValue
    FROM dbo.LabelDevice ld
    JOIN dbo.LabelType lt
      ON lt.Id = ld.LabelTypeId
)
"""
        labels_join = """
LEFT JOIN labels l
  ON l.DeviceId = d.DeviceId
"""
        labels_select = """,
    l.LabelName,
    l.LabelValue
"""

    sql = f"""
;WITH base AS (
    SELECT
        -- =====================================================================
        -- DEVICE IDENTITY
        -- =====================================================================
        d.DeviceId,
        d.DevId,
        d.DevName,
        d.TypeId,
        d.DeviceKindId,
        d.SerialNumber,
        d.IMEI,
        d.MEID,
        d.UDID,

        -- =====================================================================
        -- DEVICE STATUS & CONNECTIVITY
        -- =====================================================================
        d.Online,
        d.Mode,
        d.Flags,
        d.StatusMessage,
        d.ComplianceState,
        d.ComplianceStateDetails,

        -- =====================================================================
        -- TIMESTAMPS (Critical for temporal features)
        -- =====================================================================
        d.EnrollmentTime,
        d.LastConnTime,
        d.LastDisconnTime,
        d.LastCheckInTime,
        d.LastUpdateStatus,
        d.LastPolicyUpdateTime,
        d.LastLocationTime,
        d.LastInventoryTime,
        d.CreatedTime,
        d.ModifiedTime,

        -- =====================================================================
        -- AGENT INFO (MDM health indicator)
        -- =====================================================================
        d.AgentVersion,
        d.AgentVersionInt,
        d.AgentVersionFourPart,
        d.AgentBuildNumber,
        d.AgentMode,
        d.AgentStatus,

        -- =====================================================================
        -- HARDWARE & PLATFORM
        -- =====================================================================
        d.Manufacturer,
        d.Model,
        d.ModelNumber,
        d.ProductName,
        d.HardwareId,

        -- =====================================================================
        -- OS & FIRMWARE
        -- =====================================================================
        d.OSVersion,
        d.OSVersionInt,
        d.OSEditionId,
        d.OSBuildNumber,
        d.AndroidApiLevel,
        d.OEMVersion,
        d.FirmwareVersion,
        d.KernelVersion,
        d.BasebandVersion,
        d.BootloaderVersion,

        -- =====================================================================
        -- MEMORY & STORAGE (Critical for anomaly detection)
        -- =====================================================================
        d.TotalRAM,
        d.AvailableRAM,
        d.TotalStorage,
        d.AvailableStorage,
        d.TotalExternalStorage,
        d.AvailableExternalStorage,
        d.TotalSDCardStorage,
        d.AvailableSDCardStorage,
        d.TotalInternalStorage,
        d.AvailableInternalStorage,

        -- =====================================================================
        -- BATTERY STATUS
        -- =====================================================================
        d.BatteryStatus,
        d.BatteryLevel,
        d.BatteryHealth,
        d.BackupBatteryStatus,
        d.IsCharging,
        d.BatteryTemperature,

        -- =====================================================================
        -- SECURITY & COMPLIANCE (High value for anomaly detection)
        -- =====================================================================
        d.HasPasscode,
        d.PasscodeCompliant,
        d.IsEncrypted,
        d.EncryptionStatus,
        d.SecurityStatus,
        d.SecurityPatchLevel,
        d.IsRooted,
        d.IsJailbroken,
        d.IsDeveloperModeEnabled,
        d.IsUSBDebuggingEnabled,
        d.IsAndroidSafetynetAttestationPassed,
        d.SafetynetAttestationTime,
        d.KnoxCapability,
        d.KnoxAttestationStatus,
        d.KnoxVersion,
        d.TrustStatus,
        d.CompromisedStatus,

        -- =====================================================================
        -- NETWORK & CONNECTIVITY
        -- =====================================================================
        d.HostName,
        d.IPAddress,
        d.IPV6,
        d.MAC,
        d.WifiMAC,
        d.BluetoothMAC,
        d.WifiSSID,
        d.WifiSignalStrength,
        d.Carrier,
        d.SIMCarrierNetwork,
        d.PhoneNumber,
        d.CellularTechnology,
        d.NetworkType,
        d.InRoaming,
        d.InRoamingSIM2,
        d.DataRoamingEnabled,
        d.VoiceRoamingEnabled,
        d.IsHotspotEnabled,
        d.VPNConnected,

        -- =====================================================================
        -- LOCATION (if available)
        -- =====================================================================
        d.Latitude,
        d.Longitude,
        d.LocationAccuracy,
        d.Altitude,
        d.LocationSource,

        -- =====================================================================
        -- USER & OWNERSHIP
        -- =====================================================================
        d.CurrentPersonId,
        d.OwnershipType,
        d.AssignedUserId,
        d.UserName,
        d.UserEmail,

        -- =====================================================================
        -- DEVICE GROUPS & POLICIES
        -- =====================================================================
        d.DeviceGroupId,
        d.PolicyId,
        d.ProfileId,
        d.ConfigurationProfileCount,

        -- =====================================================================
        -- APP MANAGEMENT
        -- =====================================================================
        d.InstalledAppCount,
        d.ManagedAppCount,
        d.PendingAppCount,
        d.BlockedAppCount

    FROM dbo.DevInfo d
    WHERE d.LastCheckInTime >= :start_dt
      AND d.LastCheckInTime <  :end_dt
      {device_filter_clause}
)
{labels_cte}
SELECT {top_clause}
    b.*,

    -- =====================================================================
    -- iOS SPECIFIC FEATURES
    -- =====================================================================
    ios.IsSupervised,
    ios.IsLostModeEnabled,
    ios.IsActivationLockEnabled,
    ios.ActivationLockBypassCode,
    ios.IsDdmEnabled,
    ios.IsDeviceLocatorEnabled,
    ios.IsDoNotDisturbEnabled,
    ios.IsCloudBackupEnabled,
    ios.LastCloudBackupTime,
    ios.IsiTunesStoreAccountActive,
    ios.IsPersonalHotspotEnabled,
    ios.CellularDataUsed,
    ios.CellularDataLimit,
    ios.VoicemailCount,
    ios.ManagementProfileUpdateTime,
    ios.ManagementProfileSigningCertificateExpiry,
    ios.DEPProfileAssigned,
    ios.DEPProfilePushed,
    ios.IsMDMRemovable,
    ios.AvailableSoftwareUpdateVersion,
    ios.IsPasswordAutofillEnabled,
    ios.IsVpnConfigurationInstalled,

    -- =====================================================================
    -- WINDOWS SPECIFIC FEATURES
    -- =====================================================================
    win.IsLocked AS WindowsIsLocked,
    win.LockStatus AS WindowsLockStatus,
    win.WifiSubnet,
    win.DomainJoined,
    win.AzureADJoined,
    win.HybridAzureADJoined,
    win.AntivirusStatus,
    win.AntivirusSignatureVersion,
    win.AntivirusLastQuickScanTime,
    win.AntivirusLastFullScanTime,
    win.LastAntivirusSyncTime,
    win.FirewallStatus,
    win.AutoUpdateStatus,
    win.LastWindowsUpdateTime,
    win.PendingRebootRequired,
    win.TPMPresent,
    win.TPMEnabled,
    win.TPMVersion,
    win.SecureBootEnabled,
    win.BitLockerStatus,
    win.OsImageDeployedTime,
    win.LastBootTime,
    win.UptimeMinutes,
    win.DefenderStatus,
    win.DefenderRealTimeProtection,
    win.WindowsUpdatePendingCount,

    -- =====================================================================
    -- MAC SPECIFIC FEATURES
    -- =====================================================================
    mac.IsAppleSilicon,
    mac.IsActivationLockSupported,
    mac.IsActivationLockEnabled AS MacActivationLockEnabled,
    mac.IsContentCachingEnabled,
    mac.ContentCachingSize,
    mac.IsRemoteDesktopEnabled,
    mac.IsScreenSharingEnabled,
    mac.IsFileVaultEnabled,
    mac.BootstrapTokenEscrowed,
    mac.IsRemoteLoginEnabled,
    mac.AutoLoginEnabled,
    mac.GuestAccountEnabled,

    -- =====================================================================
    -- MAC SECURITY FEATURES
    -- =====================================================================
    macsec.IsEnabled AS FileVaultEnabled,
    macsec.FileVaultStatus,
    macsec.IsSystemIntegrityProtectionEnabled,
    macsec.IsRecoveryLockEnabled,
    macsec.RecoveryLockPasswordSet,
    macsec.FirewallEnabled AS MacFirewallEnabled,
    macsec.FirewallBlockAllIncoming,
    macsec.FirewallStealthMode,
    macsec.GatekeeperStatus,
    macsec.XProtectVersion,
    macsec.MRTVersion,

    -- =====================================================================
    -- LINUX SPECIFIC FEATURES
    -- =====================================================================
    lin.DistributionName,
    lin.DistributionVersion,
    lin.KernelVersion AS LinuxKernelVersion,
    lin.LastOSUpdateScanTime,
    lin.PendingOSUpdates,
    lin.SELinuxStatus,
    lin.AppArmorStatus,
    lin.FirewallStatus AS LinuxFirewallStatus,
    lin.SSHEnabled,
    lin.RootLoginEnabled,
    lin.PasswordAuthEnabled,

    -- =====================================================================
    -- ZEBRA ANDROID SPECIFIC FEATURES
    -- =====================================================================
    zeb.MXVersion,
    zeb.MXVersionInt,
    zeb.UserAccountsCount,
    zeb.OSXVersion,
    zeb.DeviceTrackerEnabled,
    zeb.LicenseStatus,
    zeb.LifeguardUpdateVersion,
    zeb.BatteryPartNumber,
    zeb.BatteryManufactureDate,
    zeb.BatteryCycleCount,
    zeb.BatteryDecommissionStatus,
    zeb.ScannerFirmwareVersion,
    zeb.PrinterStatus,
    zeb.WLANSignalStrength,
    zeb.EthernetMACAddress
    {labels_select}
FROM base b
LEFT JOIN dbo.iOSDevice ios
  ON ios.DevId = b.DevId
LEFT JOIN dbo.WindowsDevice win
  ON win.DeviceId = b.DeviceId
LEFT JOIN dbo.MacDevice mac
  ON mac.DeviceId = b.DeviceId
LEFT JOIN dbo.MacDeviceSecurity macsec
  ON macsec.DeviceId = b.DeviceId
LEFT JOIN dbo.LinuxDevice lin
  ON lin.DeviceId = b.DeviceId
LEFT JOIN dbo.ZebraAndroidDevice zeb
  ON zeb.DeviceId = b.DeviceId
{labels_join}
ORDER BY b.LastCheckInTime, b.DeviceId;
"""

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    logger.info(f"Loading MC inventory from {start_dt} to {end_dt}...")

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns from MC")

    # Post-process to normalize columns
    df = _normalize_mc_columns(df)

    return df


def _normalize_mc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize MC column names for consistency with feature config.

    Creates standard aliases and computes derived features.
    """
    if df.empty:
        return df

    df = df.copy()

    # Create standard column aliases
    if "OEMVersion" in df.columns:
        df["FirmwareVersion"] = df["OEMVersion"]

    if "OSVersion" in df.columns:
        df["OsVersionName"] = df["OSVersion"]

    if "Model" in df.columns:
        df["ModelName"] = df["Model"]

    # Factorize string columns to numeric codes for ML
    for col, code_col in [
        ("FirmwareVersion", "FirmwareVersionCode"),
        ("OsVersionName", "OsVersionCode"),
        ("ModelName", "ModelCode"),
        ("Manufacturer", "ManufacturerCode"),
        ("Carrier", "CarrierCode"),
    ]:
        if col in df.columns:
            df[code_col] = pd.factorize(df[col].astype(str))[0]

    # Compute derived features

    # Storage utilization
    if "TotalStorage" in df.columns and "AvailableStorage" in df.columns:
        total = df["TotalStorage"].fillna(0)
        avail = df["AvailableStorage"].fillna(0)
        df["StorageUtilizationPct"] = ((total - avail) / (total + 1)) * 100

    if "TotalRAM" in df.columns and "AvailableRAM" in df.columns:
        total = df["TotalRAM"].fillna(0)
        avail = df["AvailableRAM"].fillna(0)
        df["RAMUtilizationPct"] = ((total - avail) / (total + 1)) * 100

    # Security score (composite)
    security_cols = [
        "HasPasscode", "IsEncrypted", "IsAndroidSafetynetAttestationPassed",
        "FileVaultEnabled", "IsSystemIntegrityProtectionEnabled"
    ]
    security_score = pd.Series(0.0, index=df.index)
    security_count = 0
    for col in security_cols:
        if col in df.columns:
            security_score += df[col].fillna(0).astype(float)
            security_count += 1
    if security_count > 0:
        df["SecurityScore"] = security_score / security_count

    # Compliance score
    if "ComplianceState" in df.columns:
        df["IsCompliant"] = df["ComplianceState"].fillna("").str.lower().isin(
            ["compliant", "true", "1", "yes"]
        ).astype(int)

    # Convert boolean columns to int for ML
    bool_cols = [
        "Online", "IsCharging", "HasPasscode", "IsEncrypted", "IsRooted",
        "IsJailbroken", "IsDeveloperModeEnabled", "IsUSBDebuggingEnabled",
        "IsSupervised", "IsLostModeEnabled", "IsActivationLockEnabled",
        "FileVaultEnabled", "IsSystemIntegrityProtectionEnabled",
        "InRoaming", "VPNConnected", "IsHotspotEnabled"
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    return df


def load_mc_inventory_with_fallback(
    start_dt: str,
    end_dt: str,
    device_ids: Iterable[int] | None = None,
    include_labels: bool = True,
    limit: int | None = 1_000_000,
) -> pd.DataFrame:
    """
    Load MC inventory with fallback to simpler query if full query fails.
    """
    try:
        return load_mc_device_inventory_snapshot(
            start_dt=start_dt,
            end_dt=end_dt,
            device_ids=device_ids,
            include_labels=include_labels,
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Full MC query failed: {e}. Trying fallback...")
        return _load_basic_mc_inventory(start_dt, end_dt, device_ids, limit)


def _load_basic_mc_inventory(
    start_dt: str,
    end_dt: str,
    device_ids: Iterable[int] | None = None,
    limit: int | None = 1_000_000,
) -> pd.DataFrame:
    """
    Fallback loader with only basic MC columns.
    """
    engine = create_mc_engine()
    top_clause = f"TOP ({int(limit)})" if limit is not None else ""

    device_filter_clause = ""
    params: dict[str, object] = {"start_dt": start_dt, "end_dt": end_dt}
    if device_ids:
        device_filter_clause = "AND d.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    sql = f"""
SELECT {top_clause}
    d.DeviceId,
    d.DevId,
    d.DevName,
    d.TypeId,
    d.Online,
    d.Mode,
    d.Flags,
    d.EnrollmentTime,
    d.LastConnTime,
    d.LastDisconnTime,
    d.LastCheckInTime,
    d.AgentVersion,
    d.Manufacturer,
    d.Model,
    d.OSVersion,
    d.OEMVersion,
    d.TotalRAM,
    d.AvailableRAM,
    d.TotalStorage,
    d.AvailableStorage,
    d.BatteryStatus,
    d.IsCharging,
    d.HasPasscode,
    d.IsEncrypted,
    d.SecurityStatus,
    d.Carrier,
    d.InRoaming
FROM dbo.DevInfo d
WHERE d.LastCheckInTime >= :start_dt
  AND d.LastCheckInTime <  :end_dt
  {device_filter_clause}
ORDER BY d.LastCheckInTime, d.DeviceId;
"""

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    logger.info(f"Loaded {len(df):,} rows with basic MC columns (fallback)")
    return _normalize_mc_columns(df)


def get_available_mc_columns(table_name: str = "DevInfo") -> list[str]:
    """
    Query the database to get available columns for a MobiControl table.
    """
    engine = create_mc_engine()

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


def discover_all_mc_columns() -> dict[str, list[str]]:
    """
    Discover all columns in all relevant MC tables.
    """
    tables = [
        "DevInfo",
        "iOSDevice",
        "WindowsDevice",
        "MacDevice",
        "MacDeviceSecurity",
        "LinuxDevice",
        "ZebraAndroidDevice",
        "LabelDevice",
        "LabelType",
    ]

    result = {}
    for table in tables:
        try:
            columns = get_available_mc_columns(table)
            result[table] = columns
            logger.info(f"MC Table {table}: {len(columns)} columns")
        except Exception as e:
            logger.warning(f"Could not get MC columns for {table}: {e}")
            result[table] = []

    return result


def load_device_custom_attributes(
    device_ids: Iterable[int] | None = None,
    limit: int | None = 100_000,
) -> pd.DataFrame:
    """
    Load custom attributes for devices.

    Custom attributes are used for location mapping (Carl's requirement).
    Returns a DataFrame with DeviceId, AttributeName, AttributeValue columns
    that can be pivoted for use with LocationMapper.

    Example custom attributes for location:
        - "Store": "A101"
        - "Warehouse": "WH-North"
        - "Region": "Northeast"
    """
    engine = create_mc_engine()
    top_clause = f"TOP ({int(limit)})" if limit is not None else ""

    device_filter_clause = ""
    params: dict[str, object] = {}
    if device_ids:
        device_filter_clause = "WHERE d.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    # Query CustomAttributes table if it exists, otherwise return empty
    # Different MobiControl versions may have different schema
    sql = f"""
SELECT {top_clause}
    d.DeviceId,
    ca.Name AS AttributeName,
    ca.Value AS AttributeValue
FROM dbo.DevInfo d
INNER JOIN dbo.CustomAttribute ca
    ON ca.DeviceId = d.DeviceId
{device_filter_clause}
ORDER BY d.DeviceId, ca.Name;
"""

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        logger.info(f"Loaded {len(df):,} custom attribute records")
        return df
    except Exception as e:
        logger.warning(f"Could not load custom attributes: {e}")
        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=["DeviceId", "AttributeName", "AttributeValue"])


def get_device_custom_attributes_dict(device_ids: Iterable[int] | None = None) -> dict[int, dict[str, str]]:
    """
    Get custom attributes as a dictionary for easy lookup.

    Returns:
        Dict mapping device_id -> {attribute_name: attribute_value, ...}

    Usage with LocationMapper:
        attrs = get_device_custom_attributes_dict([123, 456])
        device_data = {"CustomAttributes": attrs.get(123, {})}
        location = mapper.get_device_location(123, device_data)
    """
    df = load_device_custom_attributes(device_ids)

    if df.empty:
        return {}

    result: dict[int, dict[str, str]] = {}
    for _, row in df.iterrows():
        device_id = row["DeviceId"]
        if device_id not in result:
            result[device_id] = {}
        result[device_id][row["AttributeName"]] = str(row["AttributeValue"])

    return result


def load_device_labels_dict(
    device_ids: Iterable[int] | None = None,
) -> dict[int, list[str]]:
    """
    Get device labels as a dictionary for easy lookup.

    Returns:
        Dict mapping device_id -> [label_value1, label_value2, ...]

    Labels can be used for location mapping with LocationMapper.
    """
    engine = create_mc_engine()

    device_filter_clause = ""
    params: dict[str, object] = {}
    if device_ids:
        device_filter_clause = "WHERE ld.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    sql = f"""
SELECT
    ld.DeviceId,
    CONVERT(nvarchar(4000), ld.Value) AS LabelValue
FROM dbo.LabelDevice ld
{device_filter_clause}
ORDER BY ld.DeviceId;
"""

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        result: dict[int, list[str]] = {}
        for _, row in df.iterrows():
            device_id = row["DeviceId"]
            if device_id not in result:
                result[device_id] = []
            result[device_id].append(str(row["LabelValue"]))

        logger.info(f"Loaded labels for {len(result)} devices")
        return result
    except Exception as e:
        logger.warning(f"Could not load device labels: {e}")
        return {}


def enrich_devices_for_location_mapping(
    devices_df: pd.DataFrame,
    include_custom_attributes: bool = True,
    include_labels: bool = True,
) -> pd.DataFrame:
    """
    Enrich a devices DataFrame with data needed for LocationMapper.

    Adds columns:
        - CustomAttributes: dict of custom attribute key-values
        - LabelDevice: list of label strings

    The enriched DataFrame can be passed directly to LocationMapper.bulk_map_devices().
    """
    if devices_df.empty:
        return devices_df

    df = devices_df.copy()
    device_ids = df["DeviceId"].unique().tolist() if "DeviceId" in df.columns else None

    # Add CustomAttributes column
    if include_custom_attributes:
        attrs_dict = get_device_custom_attributes_dict(device_ids)
        df["CustomAttributes"] = df["DeviceId"].apply(lambda x: attrs_dict.get(x, {}))
    else:
        df["CustomAttributes"] = [{}] * len(df)

    # Add LabelDevice column
    if include_labels:
        labels_dict = load_device_labels_dict(device_ids)
        df["LabelDevice"] = df["DeviceId"].apply(lambda x: labels_dict.get(x, []))
    else:
        df["LabelDevice"] = [[]] * len(df)

    return df


# =============================================================================
# Reboot/Restart Event Extraction
# =============================================================================


def load_reboot_events(
    start_dt: str,
    end_dt: str,
    device_ids: Iterable[int] | None = None,
    limit: int | None = 100_000,
) -> pd.DataFrame:
    """
    Extract reboot/restart events from MobiControl MainLog table.

    Identifies reboots via:
    1. EventClass containing reboot/restart keywords
    2. ResTxt containing device restarted messages

    Args:
        start_dt: Start datetime for event search
        end_dt: End datetime for event search
        device_ids: Optional list of device IDs to filter
        limit: Maximum rows to return

    Returns:
        DataFrame with columns: DeviceId, reboot_time, EventClass, reboot_reason, Severity
    """
    engine = create_mc_engine()
    top_clause = f"TOP ({int(limit)})" if limit is not None else ""

    device_filter_clause = ""
    params: dict[str, object] = {"start_dt": start_dt, "end_dt": end_dt}
    if device_ids:
        device_filter_clause = "AND ml.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    sql = f"""
SELECT {top_clause}
    ml.DeviceId,
    ml.DateTime AS reboot_time,
    ml.EventClass,
    ml.ResTxt AS reboot_reason,
    ml.Severity
FROM dbo.MainLog ml
WHERE ml.DateTime >= :start_dt
  AND ml.DateTime < :end_dt
  AND (
      LOWER(ml.EventClass) LIKE '%reboot%'
      OR LOWER(ml.EventClass) LIKE '%restart%'
      OR LOWER(ml.EventClass) LIKE '%boot%'
      OR LOWER(ml.EventClass) LIKE '%power cycle%'
      OR LOWER(ml.EventClass) LIKE '%device started%'
      OR LOWER(ml.ResTxt) LIKE '%device restarted%'
      OR LOWER(ml.ResTxt) LIKE '%system reboot%'
      OR LOWER(ml.ResTxt) LIKE '%device rebooted%'
      OR LOWER(ml.ResTxt) LIKE '%power on%'
  )
  {device_filter_clause}
ORDER BY ml.DeviceId, ml.DateTime;
"""

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        logger.info(f"Loaded {len(df):,} reboot events from MC MainLog")
        return df
    except Exception as e:
        logger.warning(f"Could not load reboot events: {e}")
        return pd.DataFrame(columns=["DeviceId", "reboot_time", "EventClass", "reboot_reason", "Severity"])


def calculate_reboots_from_boot_time(
    start_dt: str,
    end_dt: str,
    device_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Detect reboots by analyzing changes in LastBootTime for Windows devices.

    When LastBootTime changes between consecutive snapshots, a reboot occurred.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        device_ids: Optional device filter

    Returns:
        DataFrame with DeviceId, reboot_time (approximated), detected_from columns
    """
    engine = create_mc_engine()

    device_filter_clause = ""
    params: dict[str, object] = {"start_dt": start_dt, "end_dt": end_dt}
    if device_ids:
        device_filter_clause = "AND d.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    # Query Windows devices with LastBootTime changes
    # This uses a window function to detect when LastBootTime changes
    sql = f"""
WITH boot_times AS (
    SELECT
        d.DeviceId,
        d.LastCheckInTime AS snapshot_time,
        w.LastBootTime,
        LAG(w.LastBootTime) OVER (PARTITION BY d.DeviceId ORDER BY d.LastCheckInTime) AS prev_boot_time
    FROM dbo.DevInfo d
    JOIN dbo.WindowsDevice w ON w.DeviceId = d.DeviceId
    WHERE d.LastCheckInTime >= :start_dt
      AND d.LastCheckInTime < :end_dt
      AND w.LastBootTime IS NOT NULL
      {device_filter_clause}
)
SELECT
    DeviceId,
    snapshot_time AS detected_at,
    LastBootTime AS reboot_time,
    'boot_time_change' AS detected_from
FROM boot_times
WHERE LastBootTime != prev_boot_time
  AND prev_boot_time IS NOT NULL
ORDER BY DeviceId, reboot_time;
"""

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        logger.info(f"Detected {len(df):,} reboots from LastBootTime changes")
        return df
    except Exception as e:
        logger.warning(f"Could not detect reboots from boot time: {e}")
        return pd.DataFrame(columns=["DeviceId", "detected_at", "reboot_time", "detected_from"])


def load_all_reboot_events(
    start_dt: str,
    end_dt: str,
    device_ids: Iterable[int] | None = None,
    limit: int | None = 100_000,
) -> pd.DataFrame:
    """
    Load reboot events from all available sources and combine.

    Sources:
    1. MainLog events with reboot keywords
    2. LastBootTime changes (Windows devices)

    Returns combined DataFrame with:
        DeviceId, reboot_time, source (event_log or boot_time_change)
    """
    # Load from MainLog
    event_df = load_reboot_events(start_dt, end_dt, device_ids, limit)
    if not event_df.empty:
        event_df["source"] = "event_log"
        event_df = event_df.rename(columns={"reboot_time": "reboot_time"})
        event_df = event_df[["DeviceId", "reboot_time", "source"]]

    # Load from boot time changes
    boot_df = calculate_reboots_from_boot_time(start_dt, end_dt, device_ids)
    if not boot_df.empty:
        boot_df["source"] = "boot_time_change"
        boot_df = boot_df[["DeviceId", "reboot_time", "source"]]

    # Combine and deduplicate (events within 5 minutes are considered same reboot)
    if event_df.empty and boot_df.empty:
        return pd.DataFrame(columns=["DeviceId", "reboot_time", "source"])

    combined = pd.concat([event_df, boot_df], ignore_index=True)

    if combined.empty:
        return combined

    # Sort by device and time
    combined = combined.sort_values(["DeviceId", "reboot_time"])

    # Deduplicate: keep first event per device within 5-minute windows
    combined["reboot_time"] = pd.to_datetime(combined["reboot_time"])
    combined["time_group"] = combined.groupby("DeviceId")["reboot_time"].transform(
        lambda x: (x - x.iloc[0]).dt.total_seconds() // 300  # 5-minute groups
    )

    deduped = combined.groupby(["DeviceId", "time_group"]).first().reset_index()
    deduped = deduped[["DeviceId", "reboot_time", "source"]]

    logger.info(f"Combined {len(deduped):,} unique reboot events")
    return deduped


def aggregate_reboot_counts(
    start_dt: str,
    end_dt: str,
    device_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Aggregate reboot counts per device for the specified period.

    Returns DataFrame with:
        DeviceId, reboot_count, first_reboot, last_reboot, avg_hours_between_reboots
    """
    reboots = load_all_reboot_events(start_dt, end_dt, device_ids)

    if reboots.empty:
        return pd.DataFrame(columns=[
            "DeviceId", "reboot_count", "first_reboot", "last_reboot",
            "avg_hours_between_reboots", "consecutive_reboot_count"
        ])

    # Calculate aggregates per device
    def compute_device_reboot_stats(group: pd.DataFrame) -> pd.Series:
        """Compute reboot statistics for a single device."""
        count = len(group)
        first = group["reboot_time"].min()
        last = group["reboot_time"].max()

        # Average hours between reboots
        if count > 1:
            sorted_times = group["reboot_time"].sort_values()
            time_diffs = sorted_times.diff().dropna()
            avg_hours = time_diffs.mean().total_seconds() / 3600 if len(time_diffs) > 0 else 0
        else:
            avg_hours = 0

        # Count consecutive reboots (within 1 hour of each other)
        consecutive = 0
        if count > 1:
            sorted_times = group["reboot_time"].sort_values().reset_index(drop=True)
            for i in range(1, len(sorted_times)):
                diff_hours = (sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600
                if diff_hours < 1:
                    consecutive += 1

        return pd.Series({
            "reboot_count": count,
            "first_reboot": first,
            "last_reboot": last,
            "avg_hours_between_reboots": avg_hours,
            "consecutive_reboot_count": consecutive,
        })

    result = reboots.groupby("DeviceId").apply(compute_device_reboot_stats).reset_index()
    logger.info(f"Aggregated reboot counts for {len(result):,} devices")

    return result


# =============================================================================
# User Assignment Extraction (Carl's requirement: "People with excessive drops")
# =============================================================================


def load_device_user_assignments(
    label_types: list[str] | None = None,
    device_ids: Iterable[int] | None = None,
    limit: int | None = 100_000,
) -> pd.DataFrame:
    """
    Extract device-to-user mappings from MobiControl labels.

    Supports Carl's requirement: "People with excessive drops" by enabling
    aggregation of device abuse metrics by the assigned user.

    Args:
        label_types: Label type names to treat as user assignments.
                    Default: ["Owner", "User", "AssignedUser", "Operator"]
        device_ids: Optional device filter
        limit: Maximum rows to return

    Returns:
        DataFrame with columns:
            DeviceId, user_id, user_name, assignment_type, source_label_type
    """
    engine = create_mc_engine()
    top_clause = f"TOP ({int(limit)})" if limit is not None else ""

    # Default user assignment label types
    if label_types is None:
        label_types = ["Owner", "User", "AssignedUser", "Operator", "Employee", "Worker"]

    device_filter_clause = ""
    params: dict[str, object] = {"label_types": label_types}
    if device_ids:
        device_filter_clause = "AND ld.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    sql = f"""
SELECT {top_clause}
    ld.DeviceId,
    CONVERT(nvarchar(4000), ld.Value) AS user_id,
    CONVERT(nvarchar(4000), ld.Value) AS user_name,
    lt.Name AS assignment_type,
    lt.Name AS source_label_type
FROM dbo.LabelDevice ld
JOIN dbo.LabelType lt ON lt.Id = ld.LabelTypeId
WHERE lt.Name IN :label_types
  {device_filter_clause}
ORDER BY ld.DeviceId, lt.Name;
"""

    query = text(sql).bindparams(bindparam("label_types", expanding=True))
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        logger.info(f"Loaded {len(df):,} user assignments from MC labels")
        return df
    except Exception as e:
        logger.warning(f"Could not load user assignments from labels: {e}")
        return pd.DataFrame(columns=[
            "DeviceId", "user_id", "user_name", "assignment_type", "source_label_type"
        ])


def load_device_user_assignments_from_devinfo(
    device_ids: Iterable[int] | None = None,
    limit: int | None = 100_000,
) -> pd.DataFrame:
    """
    Extract device-to-user mappings from DevInfo table (built-in fields).

    MobiControl stores user info in DevInfo fields:
    - AssignedUserId, UserName, UserEmail, CurrentPersonId

    Args:
        device_ids: Optional device filter
        limit: Maximum rows to return

    Returns:
        DataFrame with columns:
            DeviceId, user_id, user_name, user_email, assignment_type, source
    """
    engine = create_mc_engine()
    top_clause = f"TOP ({int(limit)})" if limit is not None else ""

    device_filter_clause = ""
    params: dict[str, object] = {}
    if device_ids:
        device_filter_clause = "WHERE d.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    sql = f"""
SELECT {top_clause}
    d.DeviceId,
    d.AssignedUserId AS user_id,
    d.UserName AS user_name,
    d.UserEmail AS user_email,
    'devinfo' AS assignment_type,
    'mc_devinfo' AS source
FROM dbo.DevInfo d
{device_filter_clause}
    AND (d.AssignedUserId IS NOT NULL OR d.UserName IS NOT NULL OR d.UserEmail IS NOT NULL);
"""

    # Fix WHERE clause if no device filter
    if not device_ids:
        sql = sql.replace("AND (d.AssignedUserId", "WHERE (d.AssignedUserId")

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        logger.info(f"Loaded {len(df):,} user assignments from DevInfo")
        return df
    except Exception as e:
        logger.warning(f"Could not load user assignments from DevInfo: {e}")
        return pd.DataFrame(columns=[
            "DeviceId", "user_id", "user_name", "user_email", "assignment_type", "source"
        ])


def load_all_user_assignments(
    label_types: list[str] | None = None,
    device_ids: Iterable[int] | None = None,
    limit: int | None = 100_000,
) -> pd.DataFrame:
    """
    Load user assignments from all available sources and combine.

    Sources:
    1. LabelDevice table (user assignment labels)
    2. DevInfo table (AssignedUserId, UserName, UserEmail)

    Priority: DevInfo fields take precedence over labels if both exist.

    Args:
        label_types: Label types to treat as user assignments
        device_ids: Optional device filter
        limit: Maximum rows per source

    Returns:
        DataFrame with columns:
            DeviceId, user_id, user_name, user_email, assignment_type, source
    """
    # Load from labels
    label_df = load_device_user_assignments(label_types, device_ids, limit)
    if not label_df.empty:
        label_df["source"] = "mc_label"
        label_df["user_email"] = None  # Labels typically don't have email

    # Load from DevInfo
    devinfo_df = load_device_user_assignments_from_devinfo(device_ids, limit)

    # Combine with DevInfo taking priority
    if label_df.empty and devinfo_df.empty:
        return pd.DataFrame(columns=[
            "DeviceId", "user_id", "user_name", "user_email", "assignment_type", "source"
        ])

    # Get devices that have DevInfo assignments
    devinfo_devices = set(devinfo_df["DeviceId"].unique()) if not devinfo_df.empty else set()

    # Filter labels to only include devices without DevInfo assignments
    if not label_df.empty and devinfo_devices:
        label_df = label_df[~label_df["DeviceId"].isin(devinfo_devices)]

    # Combine
    combined = pd.concat([devinfo_df, label_df], ignore_index=True)

    # Normalize columns
    expected_cols = ["DeviceId", "user_id", "user_name", "user_email", "assignment_type", "source"]
    for col in expected_cols:
        if col not in combined.columns:
            combined[col] = None

    combined = combined[expected_cols]

    logger.info(f"Combined {len(combined):,} user assignments from all sources")
    return combined


def get_user_assignment_dict(
    label_types: list[str] | None = None,
    device_ids: Iterable[int] | None = None,
) -> dict[int, dict[str, str]]:
    """
    Get user assignments as a dictionary for easy lookup.

    Returns:
        Dict mapping device_id -> {
            "user_id": str,
            "user_name": str,
            "user_email": str | None,
            "assignment_type": str,
            "source": str,
        }
    """
    df = load_all_user_assignments(label_types, device_ids)

    if df.empty:
        return {}

    result: dict[int, dict[str, str]] = {}
    for _, row in df.iterrows():
        device_id = int(row["DeviceId"])
        result[device_id] = {
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "user_name": str(row["user_name"]) if row["user_name"] else None,
            "user_email": str(row["user_email"]) if row["user_email"] else None,
            "assignment_type": str(row["assignment_type"]) if row["assignment_type"] else "owner",
            "source": str(row["source"]) if row["source"] else "unknown",
        }

    logger.info(f"Built user assignment lookup for {len(result)} devices")
    return result


def get_device_security_summary(device_ids: Iterable[int] | None = None) -> pd.DataFrame:
    """
    Get a security-focused summary for devices.

    Returns key security indicators useful for anomaly detection.
    """
    engine = create_mc_engine()

    device_filter = ""
    params: dict[str, object] = {}
    if device_ids:
        device_filter = "WHERE d.DeviceId IN :device_ids"
        params["device_ids"] = [int(x) for x in device_ids]

    sql = f"""
SELECT
    d.DeviceId,
    d.HasPasscode,
    d.IsEncrypted,
    d.SecurityStatus,
    d.IsRooted,
    d.IsJailbroken,
    d.IsDeveloperModeEnabled,
    d.IsUSBDebuggingEnabled,
    d.IsAndroidSafetynetAttestationPassed,
    d.KnoxAttestationStatus,
    d.ComplianceState,
    d.TrustStatus,
    d.CompromisedStatus,
    ios.IsSupervised,
    ios.IsActivationLockEnabled,
    macsec.FileVaultStatus,
    macsec.IsSystemIntegrityProtectionEnabled,
    win.AntivirusStatus,
    win.FirewallStatus,
    win.BitLockerStatus
FROM dbo.DevInfo d
LEFT JOIN dbo.iOSDevice ios ON ios.DevId = d.DevId
LEFT JOIN dbo.MacDeviceSecurity macsec ON macsec.DeviceId = d.DeviceId
LEFT JOIN dbo.WindowsDevice win ON win.DeviceId = d.DeviceId
{device_filter};
"""

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    return df
