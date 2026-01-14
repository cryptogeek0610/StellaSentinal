# MobiControl & XSight Database Datapoints Reference

This document provides a comprehensive list of all datapoints that can be read from the MobiControl and XSight external data sources.

---

## Table of Contents

1. [XSight Data Warehouse](#xsight-data-warehouse)
2. [MobiControl Database](#mobicontrol-database)
3. [Summary Statistics](#summary-statistics)

---

## XSight Data Warehouse

Database: `SOTI_XSight_dw`

### 1. cs_BatteryStat (Core Battery Statistics)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `CollectedDate` | date | Collection timestamp |
| `TotalBatteryLevelDrop` | int | Battery percentage drop |
| `TotalDischargeTime_Sec` | int | Total discharge duration in seconds |
| `ChargePatternBadCount` | int | Count of bad charging patterns |
| `ChargePatternGoodCount` | int | Count of good charging patterns |
| `ChargePatternMediumCount` | int | Count of medium charging patterns |
| `AcChargeCount` | int | Number of AC charges |
| `UsbChargeCount` | int | Number of USB charges |
| `WirelessChargeCount` | int | Number of wireless charges |
| `CalculatedBatteryCapacity` | int | Calculated battery capacity (mAh) |
| `TotalFreeStorageKb` | int | Available storage in KB |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~700K

---

### 2. cs_AppUsage (App Usage Analytics)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `DeviceId` | int | Device identifier |
| `AppId` | int | Application identifier (dimension) |
| `VisitCount` | int | Number of app visits/launches |
| `TotalForegroundTime` | int | App foreground time in seconds |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~500K

---

### 3. cs_DataUsage (Network Data Consumption)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `DeviceId` | int | Device identifier |
| `AppId` | int | Application identifier (dimension) |
| `ConnectionTypeId` | int | Network type identifier: WiFi/Mobile/Roaming (dimension) |
| `Download` | bigint | Downloaded bytes by app |
| `Upload` | bigint | Uploaded bytes by app |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~300K

---

### 4. cs_BatteryAppDrain (Per-App Battery Consumption)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `DeviceId` | int | Device identifier |
| `AppId` | int | Application identifier (dimension) |
| `BatteryDrain` | int | Battery percentage drained by app |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~200K

---

### 5. cs_Heatmap (RF/Signal Quality Metrics)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `DeviceId` | int | Device identifier |
| `NetworkTypeId` | int | Network type (WiFi/Cellular/etc) (dimension) |
| `SignalStrengthBucketId` | int | Signal strength classification (dimension) |
| `ReadingCount` | int | Number of signal readings taken |
| `DropCnt` | int | Count of network drops/disconnections |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~150K

---

### 6. cs_DataUsageByHour (Hourly Data Usage)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `Hour` | int | Hour of day (0-23) (dimension) |
| `DeviceId` | int | Device identifier |
| `AppId` | int | Application identifier (dimension) |
| `ConnectionTypeId` | int | Network type (dimension) |
| `Download` | bigint | Hourly download bytes |
| `Upload` | bigint | Hourly upload bytes |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~104M (Large Table)

---

### 7. cs_BatteryLevelDrop (Hourly Battery Drain)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `Hour` | int | Hour of day (0-23) (dimension) |
| `DeviceId` | int | Device identifier |
| `BatteryLevel` | int | Battery percentage at hour |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~14.8M (Large Table)

---

### 8. cs_AppUsageListed (Hourly App Usage)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `Hour` | int | Hour of day (0-23) (dimension) |
| `DeviceId` | int | Device identifier |
| `AppId` | int | Application identifier (dimension) |
| `VisitCount` | int | Hourly app visits |
| `TotalForegroundTime` | int | Hourly foreground time in seconds |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~8.5M (Large Table)

---

### 9. cs_PresetApps (Preset App Usage)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `Hour` | int | Hour of day (0-23) (dimension) |
| `PresetAppId` | int | Preset app identifier (dimension) |
| `DeviceId` | int | Device identifier |
| `ConnectionTime` | int | Connection time in seconds |
| `Download` | bigint | Downloaded bytes |
| `Upload` | bigint | Uploaded bytes |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~19M (Large Table)

---

### 10. cs_WifiHour (WiFi Connectivity)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `Hour` | int | Hour of day (0-23) (dimension) |
| `Deviceid` | int | Device identifier (lowercase variant) |
| `AccessPointId` | int | WiFi access point identifier (dimension) |
| `WiFiSignalStrength` | int | WiFi signal strength (dBm) |
| `ConnectionTime` | int | WiFi connection duration in seconds |
| `DisconnectCount` | int | Number of WiFi disconnections |

**Timestamp Column:** `CollectedDate` | **Device Column:** `Deviceid` | **Row Count:** ~755K

---

### 11. cs_WiFiLocation (WiFi + GPS Location Data)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `Hour` | int | Hour of day (0-23) (dimension) |
| `Deviceid` | int | Device identifier |
| `AccessPointId` | int | WiFi access point identifier (dimension) |
| `ReadingTime` | datetime | Exact reading timestamp |
| `WiFiSignalStrength` | int | WiFi signal strength (dBm) |
| `Latitude` | float | GPS latitude coordinate |
| `Longitude` | float | GPS longitude coordinate |

**Timestamp Column:** `CollectedDate` | **Device Column:** `Deviceid` | **Row Count:** ~790K

---

### 12. cs_LastKnown (Last Known Location)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `DeviceId` | int | Device identifier |
| `CollectedTime` | time | Time of collection |
| `Latitude` | float | Last known latitude |
| `Longitude` | float | Last known longitude |
| `NetworkTypeId` | int | Network type when location was captured (dimension) |
| `SignalStrengthBucketId` | int | Signal strength category (dimension) |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** ~674K

---

### 13. cs_DeviceInstalledApp (App Inventory Events)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `AppId` | int | Application identifier (dimension) |
| `AppVersionId` | int | Specific app version |
| `EventTime` | datetime | Installation/removal event timestamp |
| `EventType` | varchar | Type of event (Install/Update/Remove) |

**Timestamp Column:** `EventTime` | **Device Column:** `DeviceId` | **Row Count:** ~372K

---

### 14. cs_CrashLogs (App Crash Analytics)

| Column | Type | Description |
|--------|------|-------------|
| `CollectedDate` | date | Collection date |
| `DeviceId` | int | Device identifier |
| `AppId` | int | Application identifier (dimension) |
| `CrashCount` | int | Number of crashes for the app |
| `AppNotResponding` | int | App Not Responding (ANR) count |
| `ExceptionType` | varchar | Type of exception thrown |
| `ProcessName` | varchar | Process name that crashed |

**Timestamp Column:** `CollectedDate` | **Device Column:** `DeviceId` | **Row Count:** Variable

---

## MobiControl Database

Database: `MobiControlDB`

### A. DevInfo (Core Device Inventory)

This is the primary device table containing comprehensive device information.

**Timestamp Column:** `LastCheckInTime` | **Row Count:** Millions

#### Device Identity Fields

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Primary device identifier |
| `DevId` | varchar | Device unique string ID |
| `DevName` | varchar | Device name/hostname |
| `TypeId` | int | Device type code |
| `DeviceKindId` | int | Device category identifier |
| `SerialNumber` | varchar | Hardware serial number |
| `IMEI` | varchar | International Mobile Equipment Identity |
| `MEID` | varchar | Mobile Equipment Identifier |
| `UDID` | varchar | Unique Device Identifier (iOS) |

#### Device Status & Connectivity

| Column | Type | Description |
|--------|------|-------------|
| `Online` | bit | Current online status (0/1) |
| `Mode` | varchar | Device mode status |
| `Flags` | int | Status flags bitmask |
| `StatusMessage` | varchar | Current status message |
| `ComplianceState` | varchar | Compliance status |
| `ComplianceStateDetails` | varchar | Detailed compliance info |

#### Critical Timestamps

| Column | Type | Description |
|--------|------|-------------|
| `EnrollmentTime` | datetime2 | Device enrollment timestamp |
| `LastConnTime` | datetime2 | Last successful connection |
| `LastDisconnTime` | datetime2 | Last disconnection timestamp |
| `LastCheckInTime` | datetime2 | Last check-in timestamp |
| `LastUpdateStatus` | datetime2 | Last update time |
| `LastPolicyUpdateTime` | datetime2 | Last policy sync |
| `LastLocationTime` | datetime2 | Last location update |
| `LastInventoryTime` | datetime2 | Last inventory refresh |
| `CreatedTime` | datetime2 | Record creation time |
| `ModifiedTime` | datetime2 | Record modification time |

#### MDM Agent Information

| Column | Type | Description |
|--------|------|-------------|
| `AgentVersion` | varchar | Agent version string |
| `AgentVersionInt` | int | Agent version as integer |
| `AgentVersionFourPart` | varchar | Full 4-part version |
| `AgentBuildNumber` | varchar | Build number |
| `AgentMode` | varchar | Agent operational mode |
| `AgentStatus` | varchar | Agent health status |

#### Hardware & Platform

| Column | Type | Description |
|--------|------|-------------|
| `Manufacturer` | varchar | Device manufacturer (Samsung, Apple, etc.) |
| `Model` | varchar | Device model name |
| `ModelNumber` | varchar | Model number |
| `ProductName` | varchar | Product name |
| `HardwareId` | varchar | Hardware identifier |

#### OS & Firmware Information

| Column | Type | Description |
|--------|------|-------------|
| `OSVersion` | varchar | Operating system version |
| `OSVersionInt` | int | OS version as integer |
| `OSEditionId` | int | OS edition identifier |
| `OSBuildNumber` | varchar | Build number |
| `AndroidApiLevel` | int | Android API level (Android devices) |
| `OEMVersion` | varchar | OEM-specific version |
| `FirmwareVersion` | varchar | Firmware version |
| `KernelVersion` | varchar | Kernel version |
| `BasebandVersion` | varchar | Baseband version |
| `BootloaderVersion` | varchar | Bootloader version |

#### Memory & Storage

| Column | Type | Description |
|--------|------|-------------|
| `TotalRAM` | int | Total RAM in MB |
| `AvailableRAM` | int | Available RAM in MB |
| `TotalStorage` | int | Total storage in MB |
| `AvailableStorage` | int | Available storage in MB |
| `TotalExternalStorage` | int | External SD card total storage |
| `AvailableExternalStorage` | int | External SD card available |
| `TotalSDCardStorage` | int | SD card total storage |
| `AvailableSDCardStorage` | int | SD card available |
| `TotalInternalStorage` | int | Internal storage total |
| `AvailableInternalStorage` | int | Internal storage available |

#### Battery Status

| Column | Type | Description |
|--------|------|-------------|
| `BatteryStatus` | varchar | Battery health status (Good/Fair/Poor) |
| `BatteryLevel` | int | Battery percentage (0-100) |
| `BatteryHealth` | int | Battery health code |
| `BackupBatteryStatus` | varchar | Backup battery status |
| `IsCharging` | bit | Currently charging flag |
| `BatteryTemperature` | int | Battery temperature in Celsius |

#### Security & Compliance

| Column | Type | Description |
|--------|------|-------------|
| `HasPasscode` | bit | Passcode configured |
| `PasscodeCompliant` | bit | Passcode meets policy |
| `IsEncrypted` | bit | Device encryption enabled |
| `EncryptionStatus` | varchar | Encryption status detail |
| `SecurityStatus` | varchar | Overall security status |
| `SecurityPatchLevel` | varchar | Security patch version |
| `IsRooted` | bit | Device is rooted (Android) |
| `IsJailbroken` | bit | Device is jailbroken (iOS) |
| `IsDeveloperModeEnabled` | bit | Developer mode active |
| `IsUSBDebuggingEnabled` | bit | USB debugging active |
| `IsAndroidSafetynetAttestationPassed` | bit | SafetyNet passing |
| `SafetynetAttestationTime` | datetime2 | SafetyNet check timestamp |
| `KnoxCapability` | varchar | Samsung Knox capability |
| `KnoxAttestationStatus` | varchar | Knox attestation status |
| `KnoxVersion` | varchar | Knox version |
| `TrustStatus` | varchar | Device trust status |
| `CompromisedStatus` | varchar | Compromise detection status |

#### Network & Connectivity

| Column | Type | Description |
|--------|------|-------------|
| `HostName` | varchar | Device hostname |
| `IPAddress` | varchar | Current IP address |
| `IPV6` | varchar | IPv6 address |
| `MAC` | varchar | MAC address |
| `WifiMAC` | varchar | WiFi MAC address |
| `BluetoothMAC` | varchar | Bluetooth MAC address |
| `WifiSSID` | varchar | Connected WiFi network name |
| `WifiSignalStrength` | int | WiFi signal strength (dBm) |
| `Carrier` | varchar | Mobile carrier name |
| `SIMCarrierNetwork` | varchar | SIM card carrier |
| `PhoneNumber` | varchar | Device phone number |
| `CellularTechnology` | varchar | Cellular standard (LTE/5G/etc.) |
| `NetworkType` | varchar | Current network type |
| `InRoaming` | bit | Device in roaming |
| `InRoamingSIM2` | bit | Roaming on secondary SIM |
| `DataRoamingEnabled` | bit | Data roaming enabled |
| `VoiceRoamingEnabled` | bit | Voice roaming enabled |
| `IsHotspotEnabled` | bit | Hotspot/tethering enabled |
| `VPNConnected` | bit | VPN connection active |

#### Location Information

| Column | Type | Description |
|--------|------|-------------|
| `Latitude` | float | GPS latitude |
| `Longitude` | float | GPS longitude |
| `LocationAccuracy` | int | GPS accuracy in meters |
| `Altitude` | int | GPS altitude in meters |
| `LocationSource` | varchar | Location source (GPS/WiFi/Cell/etc.) |

#### User & Ownership

| Column | Type | Description |
|--------|------|-------------|
| `CurrentPersonId` | int | Current user person ID |
| `OwnershipType` | varchar | Ownership classification (Corporate/BYOD/etc.) |
| `AssignedUserId` | int | Assigned user ID |
| `UserName` | varchar | Assigned user name |
| `UserEmail` | varchar | Assigned user email |

#### Device Groups & Policies

| Column | Type | Description |
|--------|------|-------------|
| `DeviceGroupId` | int | Device group identifier |
| `PolicyId` | int | Applied policy ID |
| `ProfileId` | int | Configuration profile ID |
| `ConfigurationProfileCount` | int | Number of config profiles |

#### App Management

| Column | Type | Description |
|--------|------|-------------|
| `InstalledAppCount` | int | Total installed apps |
| `ManagedAppCount` | int | Managed apps count |
| `PendingAppCount` | int | Apps pending installation |
| `BlockedAppCount` | int | Blocked apps count |

---

### B. iOSDevice (iOS-Specific Features)

**Join Key:** `DevId` (from DevInfo) | **Row Count:** Millions (iOS devices only)

| Column | Type | Description |
|--------|------|-------------|
| `DevId` | varchar | Device identifier |
| `IsSupervised` | bit | Device supervised enrollment |
| `IsLostModeEnabled` | bit | Find My Device enabled |
| `IsActivationLockEnabled` | bit | Activation Lock active |
| `ActivationLockBypassCode` | varchar | Bypass code if available |
| `IsDdmEnabled` | bit | Device Deployment Management |
| `IsDeviceLocatorEnabled` | bit | Device locator feature |
| `IsDoNotDisturbEnabled` | bit | Do Not Disturb mode |
| `IsCloudBackupEnabled` | bit | iCloud backup enabled |
| `LastCloudBackupTime` | datetime2 | Last backup timestamp |
| `IsiTunesStoreAccountActive` | bit | iTunes Store account active |
| `IsPersonalHotspotEnabled` | bit | Personal hotspot enabled |
| `CellularDataUsed` | bigint | Cellular data usage bytes |
| `CellularDataLimit` | bigint | Cellular data limit bytes |
| `VoicemailCount` | int | Voicemail messages count |
| `ManagementProfileUpdateTime` | datetime2 | MDM profile update time |
| `ManagementProfileSigningCertificateExpiry` | datetime2 | Cert expiration |
| `DEPProfileAssigned` | bit | Device Enrollment Program assigned |
| `DEPProfilePushed` | bit | DEP profile pushed |
| `IsMDMRemovable` | bit | MDM removable flag |
| `AvailableSoftwareUpdateVersion` | varchar | Available iOS version |
| `IsPasswordAutofillEnabled` | bit | Password autofill |
| `IsVpnConfigurationInstalled` | bit | VPN config installed |

---

### C. WindowsDevice (Windows-Specific Features)

**Join Key:** `DeviceId` | **Row Count:** Millions (Windows devices only)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `IsLocked` | bit | Device currently locked |
| `LockStatus` | varchar | Lock status detail |
| `WifiSubnet` | varchar | WiFi subnet info |
| `DomainJoined` | bit | Domain-joined flag |
| `AzureADJoined` | bit | Azure AD joined |
| `HybridAzureADJoined` | bit | Hybrid Azure AD joined |
| `AntivirusStatus` | varchar | Antivirus status (On/Off/Disabled) |
| `AntivirusSignatureVersion` | varchar | Signature version |
| `AntivirusLastQuickScanTime` | datetime2 | Last quick scan |
| `AntivirusLastFullScanTime` | datetime2 | Last full scan |
| `LastAntivirusSyncTime` | datetime2 | Last signature update |
| `FirewallStatus` | varchar | Windows Firewall status |
| `AutoUpdateStatus` | varchar | Windows Update status |
| `LastWindowsUpdateTime` | datetime2 | Last update install time |
| `PendingRebootRequired` | bit | Reboot pending flag |
| `TPMPresent` | bit | TPM chip present |
| `TPMEnabled` | bit | TPM enabled |
| `TPMVersion` | varchar | TPM version |
| `SecureBootEnabled` | bit | Secure Boot enabled |
| `BitLockerStatus` | varchar | BitLocker encryption status |
| `OsImageDeployedTime` | datetime2 | OS deployment time |
| `LastBootTime` | datetime2 | Last system boot time |
| `UptimeMinutes` | int | System uptime in minutes |
| `DefenderStatus` | varchar | Windows Defender status |
| `DefenderRealTimeProtection` | bit | Real-time protection enabled |
| `WindowsUpdatePendingCount` | int | Pending updates count |

---

### D. MacDevice (Mac-Specific Features)

**Join Key:** `DeviceId` | **Row Count:** Thousands (Mac devices)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `IsAppleSilicon` | bit | M1/M2 Apple Silicon chip |
| `IsActivationLockSupported` | bit | Activation Lock support |
| `IsActivationLockEnabled` | bit | Activation Lock enabled |
| `IsContentCachingEnabled` | bit | Content caching feature |
| `ContentCachingSize` | bigint | Cached content size |
| `IsRemoteDesktopEnabled` | bit | Remote desktop enabled |
| `IsScreenSharingEnabled` | bit | Screen sharing enabled |
| `IsFileVaultEnabled` | bit | FileVault encryption enabled |
| `BootstrapTokenEscrowed` | bit | Bootstrap token escrowed |
| `IsRemoteLoginEnabled` | bit | SSH remote login enabled |
| `AutoLoginEnabled` | bit | Auto-login feature enabled |
| `GuestAccountEnabled` | bit | Guest account enabled |

---

### E. MacDeviceSecurity (Mac Security Features)

**Join Key:** `DeviceId` | **Row Count:** Thousands (Mac devices)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `IsEnabled` | bit | FileVault encryption enabled |
| `FileVaultStatus` | varchar | FileVault status detail |
| `IsSystemIntegrityProtectionEnabled` | bit | SIP protection active |
| `IsRecoveryLockEnabled` | bit | Recovery Mode locked |
| `RecoveryLockPasswordSet` | bit | Recovery password set |
| `FirewallEnabled` | bit | Mac Firewall enabled |
| `FirewallBlockAllIncoming` | bit | Block all incoming connections |
| `FirewallStealthMode` | bit | Stealth mode enabled |
| `GatekeeperStatus` | varchar | Gatekeeper security status |
| `XProtectVersion` | varchar | XProtect definitions version |
| `MRTVersion` | varchar | Malware Removal Tool version |

---

### F. LinuxDevice (Linux-Specific Features)

**Join Key:** `DeviceId` | **Row Count:** Hundreds (Linux devices)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `DistributionName` | varchar | Linux distribution (Ubuntu/CentOS/etc.) |
| `DistributionVersion` | varchar | Distribution version |
| `KernelVersion` | varchar | Linux kernel version |
| `LastOSUpdateScanTime` | datetime2 | Last update scan |
| `PendingOSUpdates` | int | Number of pending updates |
| `SELinuxStatus` | varchar | SELinux status (Enforcing/Permissive) |
| `AppArmorStatus` | varchar | AppArmor security status |
| `FirewallStatus` | varchar | Firewall status |
| `SSHEnabled` | bit | SSH service running |
| `RootLoginEnabled` | bit | Root SSH login enabled |
| `PasswordAuthEnabled` | bit | Password authentication enabled |

---

### G. ZebraAndroidDevice (Zebra Android-Specific Features)

**Join Key:** `DeviceId` | **Row Count:** Thousands (Zebra devices)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `MXVersion` | varchar | MobileX Management System version |
| `MXVersionInt` | int | MX version as integer |
| `UserAccountsCount` | int | Number of user accounts |
| `OSXVersion` | varchar | OSX version (Zebra OS) |
| `DeviceTrackerEnabled` | bit | Device tracker feature |
| `LicenseStatus` | varchar | License status |
| `LifeguardUpdateVersion` | varchar | Lifeguard update version |
| `BatteryPartNumber` | varchar | Battery part number |
| `BatteryManufactureDate` | datetime2 | Battery manufacture date |
| `BatteryCycleCount` | int | Battery cycle count |
| `BatteryDecommissionStatus` | varchar | Battery status (Active/Retired) |
| `ScannerFirmwareVersion` | varchar | Scanner firmware version |
| `PrinterStatus` | varchar | Printer status |
| `WLANSignalStrength` | int | WiFi signal strength |
| `EthernetMACAddress` | varchar | Ethernet MAC address |

---

### H. MainLog (Activity/Event Log)

**Timestamp Column:** `DateTime` | **Device Column:** `DeviceId` | **Row Count:** ~1M

| Column | Type | Description |
|--------|------|-------------|
| `ILogId` | bigint | Log entry unique ID |
| `DateTime` | datetime2 | Event timestamp |
| `EventId` | int | Event type identifier |
| `Severity` | int | Event severity level |
| `EventClass` | varchar | Event classification/category |
| `ResTxt` | varchar | Event result/description |
| `DeviceId` | int | Associated device ID |
| `LoginId` | int | Associated login/user ID |

---

### I. Alert (System Alerts)

**Timestamp Column:** `SetDateTime` | **Row Count:** ~1.3K

| Column | Type | Description |
|--------|------|-------------|
| `AlertId` | int | Alert unique identifier |
| `AlertKey` | varchar | Alert unique key |
| `AlertName` | varchar | Alert name/title |
| `AlertSeverity` | int | Severity level |
| `DevId` | varchar | Device identifier (string) |
| `Status` | varchar | Alert status (Open/Closed/etc.) |
| `SetDateTime` | datetime2 | When alert was triggered |
| `AckDateTime` | datetime2 | When alert was acknowledged |

---

### J. DeviceStatInt (Time-Series Integer Metrics)

**Timestamp Column:** `ServerDateTime` | **Device Column:** `DeviceId` | **Row Count:** ~764K

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `TimeStamp` | datetime2 | Client-side timestamp |
| `StatType` | int | Metric type code (see mapping below) |
| `IntValue` | int | Numeric metric value |
| `ServerDateTime` | datetime2 | Server-side timestamp |

**StatType Mappings:**

| StatType | Metric Name |
|----------|-------------|
| 1 | battery_level |
| 2 | total_storage |
| 3 | available_storage |
| 4 | total_ram |
| 5 | available_ram |
| 6 | cpu_usage |
| 7 | memory_usage |
| 8 | wifi_signal_strength |
| 9 | cellular_signal_strength |
| 10 | device_temperature |
| 11 | battery_temperature |
| 12 | uptime_seconds |
| 13 | screen_on_time |
| 14 | network_rx_bytes |
| 15 | network_tx_bytes |

---

### K. DeviceStatString (Time-Series String Metrics)

**Timestamp Column:** `ServerDateTime` | **Device Column:** `DeviceId` | **Row Count:** ~349K

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `TimeStamp` | datetime2 | Client timestamp |
| `StatType` | int | String metric type code |
| `StrValue` | varchar | String metric value |
| `ServerDateTime` | datetime2 | Server timestamp |

---

### L. DeviceStatLocation (Time-Series GPS Data)

**Timestamp Column:** `ServerDateTime` | **Device Column:** `DeviceId` | **Row Count:** ~619K

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `TimeStamp` | datetime2 | Location timestamp |
| `Latitude` | float | GPS latitude |
| `Longitude` | float | GPS longitude |
| `Altitude` | int | Altitude in meters |
| `Heading` | int | Direction heading (degrees) |
| `Speed` | int | Speed in km/h |
| `ServerDateTime` | datetime2 | Server timestamp |

---

### M. DeviceStatNetTraffic (Time-Series Network Data)

**Timestamp Column:** `ServerDateTime` | **Device Column:** `DeviceId` | **Row Count:** ~244K

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `TimeStamp` | datetime2 | Traffic timestamp |
| `StatType` | int | Network metric type |
| `Upload` | bigint | Upload bytes |
| `Download` | bigint | Download bytes |
| `InterfaceType` | varchar | Network interface (WiFi/Cellular/etc.) |
| `InterfaceID` | varchar | Specific interface identifier |
| `Application` | varchar | Application name (if available) |
| `ServerDateTime` | datetime2 | Server timestamp |

---

### N. DeviceInstalledApp (App Inventory)

**Timestamp Column:** `LastChangedDate` | **Device Column:** `DeviceId` | **Row Count:** Variable

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `InstalledAppId` | int | Installed app unique ID |
| `StatusId` | int | App status code |
| `Version` | varchar | App version string |
| `Size` | int | App size in bytes |
| `DataSize` | int | App data size in bytes |
| `IsRunning` | bit | App currently running |
| `LastChangedDate` | datetime2 | Last change timestamp |

---

### O. LabelDevice + LabelType (Custom Labels/Tags)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `LabelTypeId` | int | Label type reference |
| `Value` | nvarchar | Label value (can be location, assignment, custom data) |
| `Name` | varchar | Label type name (from LabelType table) |

**Common Label Types:**
- Owner/User - Device owner/assigned user
- Location - Physical location/store/warehouse
- AssetTag - Asset tracking tag
- CostCenter - Cost center assignment
- Department - Department assignment
- Custom attributes per deployment

---

### P. CustomAttribute (Custom Device Attributes)

| Column | Type | Description |
|--------|------|-------------|
| `DeviceId` | int | Device identifier |
| `Name` | varchar | Attribute name |
| `Value` | varchar | Attribute value |

Used for location mapping and custom properties per deployment.

---

## Summary Statistics

### XSight Tables

| Table | Timestamp Column | Device Column | Row Count | Primary Use |
|-------|------------------|---------------|-----------|-------------|
| cs_BatteryStat | CollectedDate | DeviceId | ~700K | Battery health analysis |
| cs_AppUsage | CollectedDate | DeviceId | ~500K | App usage patterns |
| cs_DataUsage | CollectedDate | DeviceId | ~300K | Network usage metrics |
| cs_BatteryAppDrain | CollectedDate | DeviceId | ~200K | Per-app battery drain |
| cs_Heatmap | CollectedDate | DeviceId | ~150K | RF/Signal quality |
| cs_DataUsageByHour | CollectedDate | DeviceId | ~104M | Hourly network details |
| cs_BatteryLevelDrop | CollectedDate | DeviceId | ~14.8M | Hourly battery tracking |
| cs_AppUsageListed | CollectedDate | DeviceId | ~8.5M | Hourly app details |
| cs_PresetApps | CollectedDate | DeviceId | ~19M | Preset app usage |
| cs_WifiHour | CollectedDate | Deviceid | ~755K | WiFi connectivity |
| cs_WiFiLocation | CollectedDate | Deviceid | ~790K | WiFi + GPS location |
| cs_LastKnown | CollectedDate | DeviceId | ~674K | Last location snapshot |
| cs_DeviceInstalledApp | EventTime | DeviceId | ~372K | App inventory events |
| cs_CrashLogs | CollectedDate | DeviceId | Variable | App crashes |

### MobiControl Tables

| Table | Timestamp Column | Device Column | Row Count | Primary Use |
|-------|------------------|---------------|-----------|-------------|
| DevInfo | LastCheckInTime | DeviceId | Millions | Device inventory |
| iOSDevice | N/A | DevId | Millions | iOS-specific features |
| WindowsDevice | N/A | DeviceId | Millions | Windows-specific features |
| MacDevice | N/A | DeviceId | Thousands | Mac-specific features |
| MacDeviceSecurity | N/A | DeviceId | Thousands | Mac security details |
| LinuxDevice | N/A | DeviceId | Hundreds | Linux-specific features |
| ZebraAndroidDevice | N/A | DeviceId | Thousands | Zebra Android features |
| DeviceStatInt | ServerDateTime | DeviceId | ~764K | Integer metrics time-series |
| DeviceStatString | ServerDateTime | DeviceId | ~349K | String metrics time-series |
| DeviceStatLocation | ServerDateTime | DeviceId | ~619K | GPS coordinates time-series |
| DeviceStatNetTraffic | ServerDateTime | DeviceId | ~244K | Network traffic time-series |
| DeviceInstalledApp | LastChangedDate | DeviceId | Variable | App status tracking |
| MainLog | DateTime | DeviceId | ~1M | Activity/event log |
| Alert | SetDateTime | DevId | ~1.3K | System alerts |
| LabelDevice | N/A | DeviceId | Variable | Device labels/tags |
| CustomAttribute | N/A | DeviceId | Variable | Custom attributes |

### Totals

| Metric | Count |
|--------|-------|
| **XSight Tables** | 14 |
| **MobiControl Tables** | 16 |
| **Total Tables** | 30 |
| **Total Fields/Columns** | 300+ |

---

## Schema Discovery

The system supports dynamic schema discovery that can detect additional tables at runtime:

### XSight High-Value Patterns
- `cs_*` - All telemetry tables
- `sb_*` - Smart battery tables
- `vw_mc*` - MobiControl views
- `vw_Device*` - Device views

### MobiControl High-Value Patterns
- `DeviceStat*` - Time-series stat tables
- `Device*` - Device-related tables
- `Alert*` - Alert tables
- `Event*` - Event tables
