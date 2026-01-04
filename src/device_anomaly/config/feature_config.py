from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Set


@dataclass
class FeatureConfig:
    """
    Comprehensive feature configuration for ML anomaly detection.

    This configuration defines ALL data points that should be extracted from
    SQL databases (XSight DW, MobiControl) for training anomaly detection models.

    Feature Categories:
    - Raw features: Direct columns from SQL tables
    - Derived features: Computed from raw features (ratios, rates, etc.)
    - Rolling features: Time-series aggregations (mean, std, delta)
    - Cohort features: Z-scores relative to device cohorts
    - Interaction features: Cross-domain relationships
    """

    # =========================================================================
    # RAW FEATURES FROM XSIGHT DATA WAREHOUSE
    # =========================================================================

    # --- Battery Statistics (cs_BatteryStat) ---
    battery_features: ClassVar[List[str]] = [
        # Core battery metrics
        "TotalBatteryLevelDrop",
        "TotalDischargeTime_Sec",
        "CalculatedBatteryCapacity",
        "TotalFreeStorageKb",

        # Charge patterns
        "ChargePatternBadCount",
        "ChargePatternGoodCount",
        "ChargePatternMediumCount",

        # Charge sources
        "AcChargeCount",
        "UsbChargeCount",
        "WirelessChargeCount",

        # Battery health (extended - query if available)
        "BatteryHealth",
        "BatteryTemperature",
        "BatteryVoltage",
        "FullChargeCapacity",
        "DesignCapacity",
        "CycleCount",
        "ChargingVoltage",
        "ChargingCurrent",
        "BatteryStatus",
        "IsCharging",

        # Power management
        "ScreenOnTime_Sec",
        "ScreenOffTime_Sec",
        "PowerSaveModeTime_Sec",
        "DozeTime_Sec",
        "WakeLockTime_Sec",
        "CpuActiveTime_Sec",
        "CpuIdleTime_Sec",

        # Storage (often in BatteryStat)
        "TotalInternalStorage",
        "AvailableInternalStorage",
        "TotalExternalStorage",
        "AvailableExternalStorage",
    ]

    # --- App Usage (cs_AppUsage) ---
    app_usage_features: ClassVar[List[str]] = [
        # Core app metrics
        "AppVisitCount",
        "AppForegroundTime",
        "TotalForegroundTime",

        # Extended app metrics
        "UniqueAppsUsed",
        "SystemAppForegroundTime",
        "UserAppForegroundTime",
        "LaunchCount",
        "CrashCount",
        "ANRCount",  # App Not Responding
        "ForceStopCount",

        # Session metrics
        "LongestSessionDuration",
        "AverageSessionDuration",
        "SessionCount",

        # Background activity
        "BackgroundTime",
        "BackgroundDataUsage",
        "BackgroundBatteryDrain",

        # Notifications
        "NotificationCount",
        "NotificationDismissCount",
        "NotificationClickCount",
    ]

    # --- Web Usage (cs_AppUsage or separate) ---
    web_usage_features: ClassVar[List[str]] = [
        "WebVisitCount",
        "WebForegroundTime",
        "WebErrorCount",
        "WebPageLoadTime",
        "WebDataDownload",
        "WebDataUpload",
        "BrowserCrashCount",
        "SSLErrorCount",
        "DNSErrorCount",
    ]

    # --- Data Usage (cs_DataUsage) ---
    data_usage_features: ClassVar[List[str]] = [
        # Core data metrics
        "Download",
        "Upload",
        "TotalDownload",
        "TotalUpload",

        # By network type
        "WifiDownload",
        "WifiUpload",
        "MobileDownload",
        "MobileUpload",

        # Roaming
        "RoamingDownload",
        "RoamingUpload",

        # Background data
        "BackgroundDownload",
        "BackgroundUpload",
        "ForegroundDownload",
        "ForegroundUpload",

        # Per-app data (aggregated)
        "TopAppDataUsage",
        "SystemAppDataUsage",
    ]

    # --- Battery App Drain (cs_BatteryAppDrain) ---
    battery_drain_features: ClassVar[List[str]] = [
        "TotalBatteryAppDrain",
        "TopAppBatteryDrain",
        "SystemAppBatteryDrain",
        "UserAppBatteryDrain",
        "BackgroundAppBatteryDrain",
        "ForegroundAppBatteryDrain",
        "UniqueAppsDraining",
        "MaxSingleAppDrain",
    ]

    # --- RF/Signal Heatmap (cs_Heatmap) ---
    rf_signal_features: ClassVar[List[str]] = [
        # Signal strength
        "AvgSignalStrength",
        "MinSignalStrength",
        "MaxSignalStrength",
        "SignalStrengthStd",

        # Connection quality
        "TotalDropCnt",
        "TotalSignalReadings",
        "DropRate",

        # WiFi specific
        "WifiSignalStrength",
        "WifiDropCount",
        "WifiConnectionTime",
        "WifiDisconnectCount",

        # Cellular specific
        "CellSignalStrength",
        "CellDropCount",
        "CellConnectionTime",
        "CellTowerChanges",
        "HandoffCount",

        # Network type distribution
        "TimeOn2G",
        "TimeOn3G",
        "TimeOn4G",
        "TimeOn5G",
        "TimeOnWifi",
        "TimeOnNoNetwork",

        # Roaming
        "RoamingTime",
        "RoamingDataUsage",
    ]

    # =========================================================================
    # RAW FEATURES FROM MOBICONTROL
    # =========================================================================

    # --- Device Identity & Status ---
    mc_device_features: ClassVar[List[str]] = [
        # Identity
        "DeviceId",
        "DevId",
        "DevName",
        "TypeId",
        "DeviceKindId",

        # Status
        "Online",
        "Mode",
        "Flags",

        # Timestamps
        "EnrollmentTime",
        "LastConnTime",
        "LastDisconnTime",
        "LastCheckInTime",
        "LastUpdateStatus",

        # Agent info
        "AgentVersion",
        "AgentVersionInt",
        "AgentVersionFourPart",
    ]

    # --- Hardware & Platform ---
    mc_hardware_features: ClassVar[List[str]] = [
        "Manufacturer",
        "Model",
        "OSVersion",
        "OSVersionInt",
        "OSEditionId",
        "AndroidApiLevel",
        "OEMVersion",  # Firmware
    ]

    # --- Memory & Storage ---
    mc_memory_features: ClassVar[List[str]] = [
        "TotalRAM",
        "AvailableRAM",
        "TotalStorage",
        "AvailableStorage",
        "TotalExternalStorage",
        "AvailableExternalStorage",
        "TotalSDCardStorage",
        "AvailableSDCardStorage",
    ]

    # --- Battery Status ---
    mc_battery_features: ClassVar[List[str]] = [
        "BatteryStatus",
        "BackupBatteryStatus",
        "IsCharging",
    ]

    # --- Security & Compliance ---
    mc_security_features: ClassVar[List[str]] = [
        "HasPasscode",
        "IsEncrypted",
        "SecurityStatus",
        "IsAndroidSafetynetAttestationPassed",
        "KnoxCapability",
        "KnoxAttestationStatus",
    ]

    # --- Network ---
    mc_network_features: ClassVar[List[str]] = [
        "HostName",
        "IPV6",
        "MAC",
        "WifiMAC",
        "BluetoothMAC",
        "Carrier",
        "SIMCarrierNetwork",
        "InRoaming",
        "InRoamingSIM2",
    ]

    # --- iOS Specific ---
    mc_ios_features: ClassVar[List[str]] = [
        "IsSupervised",
        "IsLostModeEnabled",
        "IsActivationLockEnabled",
        "IsDdmEnabled",
        "ManagementProfileUpdateTime",
        "ManagementProfileSigningCertificateExpiry",
    ]

    # --- Windows Specific ---
    mc_windows_features: ClassVar[List[str]] = [
        "WindowsIsLocked",
        "WifiSubnet",
        "AntivirusLastQuickScanTime",
        "AntivirusLastFullScanTime",
        "LastAntivirusSyncTime",
        "OsImageDeployedTime",
    ]

    # --- Mac Specific ---
    mc_mac_features: ClassVar[List[str]] = [
        "IsAppleSilicon",
        "IsActivationLockSupported",
        "IsContentCachingEnabled",
        "FileVaultEnabled",
        "IsSystemIntegrityProtectionEnabled",
        "IsRecoveryLockEnabled",
    ]

    # --- Linux Specific ---
    mc_linux_features: ClassVar[List[str]] = [
        "LastOSUpdateScanTime",
    ]

    # --- Zebra Android Specific ---
    mc_zebra_features: ClassVar[List[str]] = [
        "MXVersion",
        "UserAccountsCount",
    ]

    # =========================================================================
    # CONNECTIVITY & DISCONNECT FEATURES
    # =========================================================================

    connectivity_features: ClassVar[List[str]] = [
        "DisconnectCount",
        "DisconnectFlag",
        "DisconnectWithinWindow",
        "DisconnectRecencyHours",
        "OfflineMinutes",
        "Rssi",
        "OnlineTimePct",
        "LongestOfflinePeriod",
        "OfflineEventCount",
        "ConnectionStabilityScore",
    ]

    # =========================================================================
    # LEGACY GENERIC FEATURES (backward compatibility)
    # =========================================================================

    genericFeatures: ClassVar[List[str]] = [
        "TotalBatteryLevelDrop",
        "TotalDischargeTime_Sec",
        "TotalFreeStorageKb",
        "AppVisitCount",
        "AppForegroundTime",
        "Download",
        "Upload",
        "TotalBatteryAppDrain",
        "AvgSignalStrength",
        "TotalDropCnt",
        "TotalSignalReadings",
        "DisconnectCount",
        "Rssi",
        "DisconnectFlag",
        "OfflineMinutes",
        "ChargePatternBadCount",
        "ChargePatternGoodCount",
        "ChargePatternMediumCount",
        "AcChargeCount",
        "UsbChargeCount",
        "WirelessChargeCount",
        "CalculatedBatteryCapacity",
        "FirmwareVersion",
        "OsVersionName",
        "ModelName",
        "FirmwareVersionCode",
        "OsVersionCode",
        "ModelCode",
    ]

    # =========================================================================
    # FEATURE DOMAIN MAPPING
    # =========================================================================

    feature_domains: ClassVar[Dict[str, str]] = {
        # Battery domain
        "TotalBatteryLevelDrop": "battery",
        "TotalDischargeTime_Sec": "battery",
        "TotalBatteryAppDrain": "battery",
        "CalculatedBatteryCapacity": "battery",
        "ChargePatternBadCount": "battery",
        "ChargePatternGoodCount": "battery",
        "ChargePatternMediumCount": "battery",
        "AcChargeCount": "battery",
        "UsbChargeCount": "battery",
        "WirelessChargeCount": "battery",
        "BatteryHealth": "battery",
        "BatteryTemperature": "battery",
        "BatteryVoltage": "battery",
        "FullChargeCapacity": "battery",
        "DesignCapacity": "battery",
        "CycleCount": "battery",
        "ChargingVoltage": "battery",
        "ChargingCurrent": "battery",
        "BatteryStatus": "battery",
        "IsCharging": "battery",
        "ScreenOnTime_Sec": "battery",
        "ScreenOffTime_Sec": "battery",
        "PowerSaveModeTime_Sec": "battery",
        "DozeTime_Sec": "battery",
        "WakeLockTime_Sec": "battery",
        "TopAppBatteryDrain": "battery",
        "SystemAppBatteryDrain": "battery",
        "UserAppBatteryDrain": "battery",
        "BackgroundAppBatteryDrain": "battery",
        "ForegroundAppBatteryDrain": "battery",
        "MaxSingleAppDrain": "battery",

        # RF/Network domain
        "AvgSignalStrength": "rf",
        "MinSignalStrength": "rf",
        "MaxSignalStrength": "rf",
        "SignalStrengthStd": "rf",
        "TotalDropCnt": "rf",
        "TotalSignalReadings": "rf",
        "DropRate": "rf",
        "Rssi": "rf",
        "DisconnectCount": "rf",
        "DisconnectFlag": "rf",
        "OfflineMinutes": "rf",
        "WifiSignalStrength": "rf",
        "WifiDropCount": "rf",
        "WifiConnectionTime": "rf",
        "WifiDisconnectCount": "rf",
        "CellSignalStrength": "rf",
        "CellDropCount": "rf",
        "CellConnectionTime": "rf",
        "CellTowerChanges": "rf",
        "HandoffCount": "rf",
        "TimeOn2G": "rf",
        "TimeOn3G": "rf",
        "TimeOn4G": "rf",
        "TimeOn5G": "rf",
        "TimeOnWifi": "rf",
        "TimeOnNoNetwork": "rf",
        "RoamingTime": "rf",
        "ConnectionStabilityScore": "rf",

        # Throughput domain
        "Download": "throughput",
        "Upload": "throughput",
        "TotalDownload": "throughput",
        "TotalUpload": "throughput",
        "WifiDownload": "throughput",
        "WifiUpload": "throughput",
        "MobileDownload": "throughput",
        "MobileUpload": "throughput",
        "RoamingDownload": "throughput",
        "RoamingUpload": "throughput",
        "BackgroundDownload": "throughput",
        "BackgroundUpload": "throughput",
        "ForegroundDownload": "throughput",
        "ForegroundUpload": "throughput",
        "RoamingDataUsage": "throughput",

        # Usage domain
        "AppVisitCount": "usage",
        "AppForegroundTime": "usage",
        "TotalForegroundTime": "usage",
        "UniqueAppsUsed": "usage",
        "SystemAppForegroundTime": "usage",
        "UserAppForegroundTime": "usage",
        "LaunchCount": "usage",
        "CrashCount": "usage",
        "ANRCount": "usage",
        "ForceStopCount": "usage",
        "LongestSessionDuration": "usage",
        "AverageSessionDuration": "usage",
        "SessionCount": "usage",
        "BackgroundTime": "usage",
        "NotificationCount": "usage",
        "NotificationDismissCount": "usage",
        "NotificationClickCount": "usage",
        "WebVisitCount": "usage",
        "WebForegroundTime": "usage",
        "WebErrorCount": "usage",

        # Storage domain
        "TotalFreeStorageKb": "storage",
        "TotalInternalStorage": "storage",
        "AvailableInternalStorage": "storage",
        "TotalExternalStorage": "storage",
        "AvailableExternalStorage": "storage",
        "TotalStorage": "storage",
        "AvailableStorage": "storage",
        "TotalRAM": "storage",
        "AvailableRAM": "storage",
        "TotalSDCardStorage": "storage",
        "AvailableSDCardStorage": "storage",

        # CPU domain
        "CpuActiveTime_Sec": "cpu",
        "CpuIdleTime_Sec": "cpu",

        # Security domain
        "HasPasscode": "security",
        "IsEncrypted": "security",
        "SecurityStatus": "security",
        "IsAndroidSafetynetAttestationPassed": "security",
        "KnoxCapability": "security",
        "KnoxAttestationStatus": "security",
        "IsSupervised": "security",
        "IsLostModeEnabled": "security",
        "IsActivationLockEnabled": "security",
        "FileVaultEnabled": "security",
        "IsSystemIntegrityProtectionEnabled": "security",
        "IsRecoveryLockEnabled": "security",
        "AntivirusLastQuickScanTime": "security",
        "AntivirusLastFullScanTime": "security",

        # Firmware/Hardware domain
        "FirmwareVersion": "firmware",
        "FirmwareVersionCode": "firmware",
        "OsVersionName": "firmware",
        "OsVersionId": "firmware",
        "OsVersionCode": "firmware",
        "OSVersion": "firmware",
        "OSVersionInt": "firmware",
        "OEMVersion": "firmware",
        "AndroidApiLevel": "firmware",
        "AgentVersion": "firmware",
        "AgentVersionInt": "firmware",
        "ModelName": "hardware",
        "ModelId": "hardware",
        "ModelCode": "hardware",
        "Manufacturer": "hardware",
        "Model": "hardware",
    }

    # =========================================================================
    # DOMAIN WEIGHTS FOR ML
    # =========================================================================

    domain_weights: ClassVar[Dict[str, float]] = {
        "battery": 0.6,       # Lower to prevent domination
        "rf": 1.1,            # Higher for connectivity issues
        "throughput": 1.0,
        "usage": 1.0,
        "storage": 0.9,
        "cpu": 0.8,
        "security": 1.3,      # High weight for security anomalies
        "firmware": 1.2,      # Important for cohort analysis
        "hardware": 1.0,
    }

    # =========================================================================
    # DERIVED FEATURE DEFINITIONS
    # =========================================================================

    @dataclass
    class DerivedFeature:
        """Definition of a computed feature."""
        name: str
        formula: str  # Description of calculation
        dependencies: List[str]
        domain: str

    derived_features: ClassVar[List["FeatureConfig.DerivedFeature"]] = []

    # Battery efficiency features
    derived_feature_definitions: ClassVar[Dict[str, Dict]] = {
        # Battery efficiency
        "BatteryDrainPerHour": {
            "formula": "TotalBatteryLevelDrop / (TotalDischargeTime_Sec / 3600 + 1)",
            "dependencies": ["TotalBatteryLevelDrop", "TotalDischargeTime_Sec"],
            "domain": "battery",
        },
        "BatteryDrainPerAppHour": {
            "formula": "TotalBatteryAppDrain / (AppForegroundTime / 3600 + 1)",
            "dependencies": ["TotalBatteryAppDrain", "AppForegroundTime"],
            "domain": "battery",
        },
        "BatteryDrainPerMB": {
            "formula": "TotalBatteryLevelDrop / ((Download + Upload) / 1e6 + 1)",
            "dependencies": ["TotalBatteryLevelDrop", "Download", "Upload"],
            "domain": "battery",
        },
        "ChargeQualityScore": {
            "formula": "ChargePatternGoodCount / (ChargePatternGoodCount + ChargePatternBadCount + 1)",
            "dependencies": ["ChargePatternGoodCount", "ChargePatternBadCount"],
            "domain": "battery",
        },
        "WirelessChargePreference": {
            "formula": "WirelessChargeCount / (AcChargeCount + UsbChargeCount + WirelessChargeCount + 1)",
            "dependencies": ["WirelessChargeCount", "AcChargeCount", "UsbChargeCount"],
            "domain": "battery",
        },
        "BatteryHealthRatio": {
            "formula": "CalculatedBatteryCapacity / (DesignCapacity + 1)",
            "dependencies": ["CalculatedBatteryCapacity", "DesignCapacity"],
            "domain": "battery",
        },
        "ScreenOnRatio": {
            "formula": "ScreenOnTime_Sec / (ScreenOnTime_Sec + ScreenOffTime_Sec + 1)",
            "dependencies": ["ScreenOnTime_Sec", "ScreenOffTime_Sec"],
            "domain": "battery",
        },

        # Network efficiency
        "DataPerSignalQuality": {
            "formula": "(Download + Upload) / (AvgSignalStrength + 100)",  # +100 to handle negative dBm
            "dependencies": ["Download", "Upload", "AvgSignalStrength"],
            "domain": "rf",
        },
        "DropsPerActiveHour": {
            "formula": "TotalDropCnt / (AppForegroundTime / 3600 + 1)",
            "dependencies": ["TotalDropCnt", "AppForegroundTime"],
            "domain": "rf",
        },
        "DropRate": {
            "formula": "TotalDropCnt / (TotalSignalReadings + 1)",
            "dependencies": ["TotalDropCnt", "TotalSignalReadings"],
            "domain": "rf",
        },
        "SignalVariability": {
            "formula": "(MaxSignalStrength - MinSignalStrength) / (abs(AvgSignalStrength) + 1)",
            "dependencies": ["MaxSignalStrength", "MinSignalStrength", "AvgSignalStrength"],
            "domain": "rf",
        },
        "WifiVsCellRatio": {
            "formula": "TimeOnWifi / (TimeOn4G + TimeOn5G + TimeOnWifi + 1)",
            "dependencies": ["TimeOnWifi", "TimeOn4G", "TimeOn5G"],
            "domain": "rf",
        },
        "DisconnectSeverity": {
            "formula": "DisconnectCount * OfflineMinutes",
            "dependencies": ["DisconnectCount", "OfflineMinutes"],
            "domain": "rf",
        },
        "ConnectionStabilityScore": {
            "formula": "1 - (TotalDropCnt / (TotalSignalReadings + 1))",
            "dependencies": ["TotalDropCnt", "TotalSignalReadings"],
            "domain": "rf",
        },

        # Storage utilization
        "StorageUtilization": {
            "formula": "1 - (AvailableStorage / (TotalStorage + 1))",
            "dependencies": ["AvailableStorage", "TotalStorage"],
            "domain": "storage",
        },
        "RAMPressure": {
            "formula": "1 - (AvailableRAM / (TotalRAM + 1))",
            "dependencies": ["AvailableRAM", "TotalRAM"],
            "domain": "storage",
        },
        "InternalStorageUtilization": {
            "formula": "1 - (AvailableInternalStorage / (TotalInternalStorage + 1))",
            "dependencies": ["AvailableInternalStorage", "TotalInternalStorage"],
            "domain": "storage",
        },

        # Usage intensity
        "UsageIntensity": {
            "formula": "AppForegroundTime / (24 * 3600)",  # Fraction of day
            "dependencies": ["AppForegroundTime"],
            "domain": "usage",
        },
        "AppDiversity": {
            "formula": "UniqueAppsUsed / (AppVisitCount + 1)",  # Lower = focused usage
            "dependencies": ["UniqueAppsUsed", "AppVisitCount"],
            "domain": "usage",
        },
        "CrashRate": {
            "formula": "CrashCount / (AppVisitCount + 1)",
            "dependencies": ["CrashCount", "AppVisitCount"],
            "domain": "usage",
        },
        "ANRRate": {
            "formula": "ANRCount / (AppVisitCount + 1)",
            "dependencies": ["ANRCount", "AppVisitCount"],
            "domain": "usage",
        },
        "WebErrorsPerVisit": {
            "formula": "WebErrorCount / (WebVisitCount + 1)",
            "dependencies": ["WebErrorCount", "WebVisitCount"],
            "domain": "usage",
        },
        "NotificationEngagement": {
            "formula": "NotificationClickCount / (NotificationCount + 1)",
            "dependencies": ["NotificationClickCount", "NotificationCount"],
            "domain": "usage",
        },
        "BackgroundVsForeground": {
            "formula": "BackgroundTime / (AppForegroundTime + 1)",
            "dependencies": ["BackgroundTime", "AppForegroundTime"],
            "domain": "usage",
        },

        # Data usage patterns
        "UploadToDownloadRatio": {
            "formula": "Upload / (Download + 1)",
            "dependencies": ["Upload", "Download"],
            "domain": "throughput",
        },
        "MobileToWifiDataRatio": {
            "formula": "(MobileDownload + MobileUpload) / (WifiDownload + WifiUpload + 1)",
            "dependencies": ["MobileDownload", "MobileUpload", "WifiDownload", "WifiUpload"],
            "domain": "throughput",
        },
        "BackgroundDataRatio": {
            "formula": "(BackgroundDownload + BackgroundUpload) / (Download + Upload + 1)",
            "dependencies": ["BackgroundDownload", "BackgroundUpload", "Download", "Upload"],
            "domain": "throughput",
        },
        "DataPerApp": {
            "formula": "(Download + Upload) / (UniqueAppsUsed + 1)",
            "dependencies": ["Download", "Upload", "UniqueAppsUsed"],
            "domain": "throughput",
        },

        # Cross-domain composite scores
        "DeviceHealthScore": {
            "formula": "Composite of battery + storage + connectivity",
            "dependencies": ["BatteryHealthRatio", "StorageUtilization", "ConnectionStabilityScore"],
            "domain": "composite",
        },
        "AnomalyRiskScore": {
            "formula": "Weighted sum of anomaly indicators",
            "dependencies": ["CrashRate", "DisconnectSeverity", "BatteryDrainPerHour"],
            "domain": "composite",
        },
    }

    # =========================================================================
    # ROLLING WINDOW FEATURE DEFINITIONS
    # =========================================================================

    rolling_windows: ClassVar[List[int]] = [7, 14, 30]  # Days

    rolling_aggregations: ClassVar[List[str]] = [
        "mean",
        "std",
        "min",
        "max",
        "median",
    ]

    # Features to compute rolling statistics for
    rolling_feature_candidates: ClassVar[List[str]] = [
        "TotalBatteryLevelDrop",
        "TotalDischargeTime_Sec",
        "AppForegroundTime",
        "AppVisitCount",
        "Download",
        "Upload",
        "TotalBatteryAppDrain",
        "AvgSignalStrength",
        "TotalDropCnt",
        "DisconnectCount",
        "OfflineMinutes",
        "TotalFreeStorageKb",
        "CrashCount",
        "ANRCount",
        "WebErrorCount",
    ]

    # =========================================================================
    # TEMPORAL FEATURES
    # =========================================================================

    temporal_features: ClassVar[List[str]] = [
        "hour_of_day",
        "hour_of_day_norm",
        "day_of_week",
        "day_of_week_norm",
        "is_weekend",
        "is_business_hours",  # 9-17
        "day_of_month",
        "week_of_year",
        "month",
        "quarter",
    ]

    # =========================================================================
    # COHORT GROUPING COLUMNS
    # =========================================================================

    cohort_columns: ClassVar[List[str]] = [
        "ManufacturerId",
        "ModelId",
        "OsVersionId",
        "FirmwareVersion",
    ]

    # =========================================================================
    # FEATURE EXCLUSIONS
    # =========================================================================

    # Columns to never use as ML features
    excluded_columns: ClassVar[Set[str]] = {
        "DeviceId",
        "DevId",
        "DevName",
        "Timestamp",
        "CollectedDate",
        "tenant_id",
        "cohort_id",
        "is_injected_anomaly",
        "anomaly_label",
        "anomaly_score",
        "HostName",
        "IPV6",
        "MAC",
        "WifiMAC",
        "BluetoothMAC",
        "CurrentPersonId",
        "LabelName",
        "LabelValue",
    }

    # =========================================================================
    # CEO REQUIREMENTS: SHIFT-AWARE FEATURES (Carl's Requirements)
    # =========================================================================

    shift_aware_features: ClassVar[Dict[str, Dict]] = {
        # Battery shift features
        "BatteryAtShiftStart": {
            "formula": "BatteryLevel at shift start time",
            "dependencies": ["BatteryLevel", "ShiftStartTime"],
            "domain": "battery",
        },
        "BatteryAtShiftEnd": {
            "formula": "Projected BatteryLevel at shift end",
            "dependencies": ["BatteryLevel", "BatteryDrainPerHour", "ShiftDurationHours"],
            "domain": "battery",
        },
        "ShiftCompletionProbability": {
            "formula": "Probability device lasts full shift",
            "dependencies": ["BatteryLevel", "BatteryDrainPerHour", "ShiftDurationHours"],
            "domain": "battery",
        },
        "HoursUntilBatteryDead": {
            "formula": "BatteryLevel / BatteryDrainPerHour",
            "dependencies": ["BatteryLevel", "BatteryDrainPerHour"],
            "domain": "battery",
        },
        "WasFullyChargedMorning": {
            "formula": "1 if BatteryLevel >= 90 at 6AM-8AM, else 0",
            "dependencies": ["BatteryLevel", "Timestamp"],
            "domain": "battery",
        },
        "ShiftDrainTotal": {
            "formula": "Battery drain during shift hours",
            "dependencies": ["BatteryLevel", "ShiftStartTime", "ShiftEndTime"],
            "domain": "battery",
        },

        # Network pattern features (Carl: AP hopping, carrier patterns)
        "ApChangeCount": {
            "formula": "Number of unique APs connected in period",
            "dependencies": ["UniqueAPsConnected"],
            "domain": "rf",
        },
        "ApChangeRate": {
            "formula": "AP changes per hour of WiFi time",
            "dependencies": ["UniqueAPsConnected", "TimeOnWifi"],
            "domain": "rf",
        },
        "ApStickinessScore": {
            "formula": "1 - (ApChangeRate normalized), higher = more sticky",
            "dependencies": ["ApChangeRate"],
            "domain": "rf",
        },
        "CellTowerChangeCount": {
            "formula": "Number of cell tower changes",
            "dependencies": ["UniqueCellIds"],
            "domain": "rf",
        },
        "CellTowerChangeRate": {
            "formula": "Tower changes per hour of cellular time",
            "dependencies": ["UniqueCellIds", "TimeOn4G", "TimeOn5G"],
            "domain": "rf",
        },
        "DisconnectPredictability": {
            "formula": "Variance in disconnect timing (lower = more predictable)",
            "dependencies": ["DisconnectCount", "Timestamp"],
            "domain": "rf",
        },
        "NetworkTypeDiversity": {
            "formula": "Count of different network types used",
            "dependencies": ["NetworkTypeCount"],
            "domain": "rf",
        },
        "TechFallbackRate": {
            "formula": "(TimeOn3G + TimeOn2G) / (TimeOn5G + TimeOn4G + TimeOn3G + TimeOn2G + 1)",
            "dependencies": ["TimeOn5G", "TimeOn4G", "TimeOn3G", "TimeOn2G"],
            "domain": "rf",
        },

        # App power features (Carl: Apps consuming too much power)
        "AppDrainPerForegroundHour": {
            "formula": "TotalBatteryAppDrain / (AppForegroundTime / 3600 + 1)",
            "dependencies": ["TotalBatteryAppDrain", "AppForegroundTime"],
            "domain": "battery",
        },
        "BackgroundDrainRatio": {
            "formula": "BackgroundAppBatteryDrain / (TotalBatteryAppDrain + 1)",
            "dependencies": ["BackgroundAppBatteryDrain", "TotalBatteryAppDrain"],
            "domain": "battery",
        },
        "TopAppDrainContribution": {
            "formula": "TopAppBatteryDrain / (TotalBatteryAppDrain + 1)",
            "dependencies": ["TopAppBatteryDrain", "TotalBatteryAppDrain"],
            "domain": "battery",
        },
        "CrashCountNormalized": {
            "formula": "CrashCount / days_in_period",
            "dependencies": ["CrashCount"],
            "domain": "usage",
        },
        "ANRCountNormalized": {
            "formula": "ANRCount / days_in_period",
            "dependencies": ["ANRCount"],
            "domain": "usage",
        },

        # Device abuse features (Carl: Excessive drops/reboots)
        "DropCountNormalized": {
            "formula": "TotalDropCnt / days_in_period",
            "dependencies": ["TotalDropCnt"],
            "domain": "rf",
        },
        "RebootCountNormalized": {
            "formula": "RebootCount / days_in_period",
            "dependencies": ["RebootCount"],
            "domain": "usage",
        },
        "UserDropPattern": {
            "formula": "Consistency of drops by user across devices",
            "dependencies": ["TotalDropCnt", "UserId"],
            "domain": "rf",
        },
        "LocationDropPattern": {
            "formula": "Consistency of drops at location across devices",
            "dependencies": ["TotalDropCnt", "LocationId"],
            "domain": "rf",
        },
        "AbuseScore": {
            "formula": "Composite of drops + reboots + crashes",
            "dependencies": ["TotalDropCnt", "RebootCount", "CrashCount"],
            "domain": "composite",
        },

        # Location comparison features (Carl: Warehouse 1 vs Warehouse 2)
        "LocationAnomalyRate": {
            "formula": "Anomalies at location / total devices at location",
            "dependencies": ["AnomalyCount", "DeviceCount"],
            "domain": "location",
        },
        "VsLocationBaseline": {
            "formula": "Current metric / location baseline",
            "dependencies": ["CurrentValue", "BaselineValue"],
            "domain": "location",
        },
        "LocationRank": {
            "formula": "Rank among all locations for key metrics",
            "dependencies": ["MetricValue"],
            "domain": "location",
        },
    }

    # =========================================================================
    # CEO REQUIREMENTS: HEURISTIC RULES (Shift-Aware)
    # =========================================================================

    heuristic_rules: ClassVar[List[Dict]] = [
        # Battery shift failures (Carl: "Batteries not lasting a shift")
        {
            "name": "battery_shift_failure_risk",
            "column": "BatteryDrainPerHour",
            "threshold": 12.5,
            "op": ">=",
            "severity": 0.8,
            "description": "Battery drain rate too high to complete 8-hour shift",
        },
        {
            "name": "low_battery_at_shift_start",
            "column": "BatteryLevel",
            "threshold": 80,
            "op": "<",
            "severity": 0.6,
            "description": "Battery not fully charged at shift start",
        },
        {
            "name": "critical_battery_during_shift",
            "column": "BatteryLevel",
            "threshold": 20,
            "op": "<",
            "severity": 0.9,
            "description": "Critical battery level during work hours",
        },

        # Charging patterns (Carl: "Batteries not fully charged in the morning")
        {
            "name": "morning_charge_incomplete",
            "column": "WasFullyChargedMorning",
            "threshold": 1,
            "op": "<",
            "severity": 0.5,
            "description": "Device not fully charged at morning shift start",
        },
        {
            "name": "poor_charging_patterns",
            "column": "ChargePatternBadCount",
            "threshold": 3,
            "op": ">=",
            "severity": 0.5,
            "description": "Frequent poor charging patterns",
        },

        # AP hopping (Carl: "AP hopping/stickiness")
        {
            "name": "excessive_ap_roaming",
            "column": "UniqueAPsConnected",
            "threshold": 10,
            "op": ">=",
            "severity": 0.6,
            "description": "Excessive WiFi AP switching",
        },
        {
            "name": "wifi_stickiness_weak_signal",
            "column": "WifiSignalStrength",
            "threshold": -75,
            "op": "<",
            "severity": 0.5,
            "description": "Device stuck on weak WiFi AP",
        },

        # Tower hopping (Carl: "Tower hopping/stickiness")
        {
            "name": "excessive_tower_hopping",
            "column": "UniqueCellIds",
            "threshold": 5,
            "op": ">=",
            "severity": 0.5,
            "description": "Excessive cell tower switching",
        },

        # Disconnects (Carl: "Server disconnection patterns")
        {
            "name": "high_disconnect_rate",
            "column": "WifiDisconnectCount",
            "threshold": 10,
            "op": ">=",
            "severity": 0.7,
            "description": "High WiFi disconnection rate",
        },
        {
            "name": "excessive_offline_time",
            "column": "TimeOnNoNetwork",
            "threshold": 120,
            "op": ">=",
            "severity": 0.6,
            "description": "Excessive time without network connectivity",
        },

        # Drops (Carl: "Devices with excessive drops")
        {
            "name": "excessive_drops",
            "column": "TotalDropCnt",
            "threshold": 5,
            "op": ">=",
            "severity": 0.7,
            "description": "Excessive physical device drops",
        },
        {
            "name": "high_drop_rate",
            "column": "DropRate",
            "threshold": 0.1,
            "op": ">=",
            "severity": 0.6,
            "description": "High drop rate relative to usage",
        },

        # Reboots (Carl: "Devices with excessive reboots")
        {
            "name": "excessive_reboots",
            "column": "RebootCount",
            "threshold": 3,
            "op": ">=",
            "severity": 0.7,
            "description": "Excessive device reboots",
        },

        # App crashes (Carl: "Crashes")
        {
            "name": "high_crash_rate",
            "column": "CrashCount",
            "threshold": 5,
            "op": ">=",
            "severity": 0.7,
            "description": "High application crash rate",
        },
        {
            "name": "anr_issues",
            "column": "ANRCount",
            "threshold": 3,
            "op": ">=",
            "severity": 0.6,
            "description": "Frequent App Not Responding events",
        },

        # App power drain (Carl: "Apps consuming too much power")
        {
            "name": "high_app_battery_drain",
            "column": "TotalBatteryAppDrain",
            "threshold": 50,
            "op": ">=",
            "severity": 0.6,
            "description": "Apps consuming excessive battery",
        },
        {
            "name": "single_app_drain_spike",
            "column": "TopAppBatteryDrain",
            "threshold": 30,
            "op": ">=",
            "severity": 0.7,
            "description": "Single app consuming too much battery",
        },

        # Cohort issues (Carl: "Performance by manufacturer/model/OS")
        {
            "name": "storage_pressure",
            "column": "StorageUtilization",
            "threshold": 0.9,
            "op": ">=",
            "severity": 0.6,
            "description": "Storage almost full",
        },
        {
            "name": "memory_pressure",
            "column": "RAMPressure",
            "threshold": 0.85,
            "op": ">=",
            "severity": 0.6,
            "description": "High memory pressure",
        },
    ]

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @classmethod
    def get_all_raw_features(cls) -> List[str]:
        """Get all raw feature names from all categories."""
        features = []
        features.extend(cls.battery_features)
        features.extend(cls.app_usage_features)
        features.extend(cls.web_usage_features)
        features.extend(cls.data_usage_features)
        features.extend(cls.battery_drain_features)
        features.extend(cls.rf_signal_features)
        features.extend(cls.connectivity_features)
        features.extend(cls.mc_memory_features)
        features.extend(cls.mc_security_features)
        features.extend(cls.mc_ios_features)
        features.extend(cls.mc_windows_features)
        features.extend(cls.mc_mac_features)
        features.extend(cls.mc_zebra_features)
        return list(set(features))

    @classmethod
    def get_domain_for_feature(cls, feature_name: str) -> str:
        """Get the domain category for a feature."""
        # Check direct mapping
        if feature_name in cls.feature_domains:
            return cls.feature_domains[feature_name]

        # Check derived features
        if feature_name in cls.derived_feature_definitions:
            return cls.derived_feature_definitions[feature_name].get("domain", "unknown")

        # Infer from suffixes
        if "_roll_" in feature_name or "_delta" in feature_name or "_cohort_z" in feature_name:
            base_name = feature_name.split("_roll_")[0].split("_delta")[0].split("_cohort_z")[0]
            return cls.feature_domains.get(base_name, "derived")

        return "unknown"

    @classmethod
    def get_weight_for_feature(cls, feature_name: str) -> float:
        """Get the ML weight for a feature based on its domain."""
        domain = cls.get_domain_for_feature(feature_name)
        return cls.domain_weights.get(domain, 1.0)

    @classmethod
    def get_derived_feature_names(cls) -> List[str]:
        """Get all derived feature names."""
        return list(cls.derived_feature_definitions.keys())

    @classmethod
    def get_features_for_domain(cls, domain: str) -> List[str]:
        """Get all features belonging to a specific domain."""
        return [f for f, d in cls.feature_domains.items() if d == domain]

    @classmethod
    def get_shift_aware_features(cls) -> List[str]:
        """Get all shift-aware feature names."""
        return list(cls.shift_aware_features.keys())

    @classmethod
    def get_heuristic_rules(cls) -> List[Dict]:
        """Get all heuristic rule configurations."""
        return cls.heuristic_rules

    @classmethod
    def get_heuristic_rules_for_domain(cls, domain: str) -> List[Dict]:
        """Get heuristic rules for a specific domain."""
        domain_columns = cls.get_features_for_domain(domain)
        return [r for r in cls.heuristic_rules if r.get("column") in domain_columns]

    @classmethod
    def get_all_insight_features(cls) -> List[str]:
        """Get all features relevant for CEO insight requirements."""
        features = []
        features.extend(cls.get_derived_feature_names())
        features.extend(cls.get_shift_aware_features())
        return list(set(features))
