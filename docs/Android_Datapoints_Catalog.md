# Android Datapoints Catalog
## Telemetry Agent Data Collection Guide

**Target Platform:** Android 8.0 (API 26) and above  
**Management Context:** Fully Managed Device Owner  
**Document Version:** 1.0  
**Last Updated:** 2024

---

## Table of Contents

1. [Overview](#overview)
2. [Taxonomy of Collectible Datapoints](#taxonomy-of-collectible-datapoints)
3. [Detailed Datapoint Specifications](#detailed-datapoint-specifications)
4. [Implementation Guidance](#implementation-guidance)
5. [Safe MVP Collector List](#safe-mvp-collector-list)
6. [Device Owner Enhanced List](#device-owner-enhanced-list)
7. [Red List - Infeasible/Restricted Datapoints](#red-list---infeasiblerestricted-datapoints)
8. [References and Citations](#references-and-citations)

---

## Overview

This catalog documents Android APIs available for collecting device telemetry data in a fully managed (Device Owner) context, targeting Android 8.0 (API level 26) as the minimum version. The catalog is organized by datapoint category, with detailed specifications for each collectible metric.

### Key Assumptions

- **Minimum Android Version:** Android 8.0 (API 26)
- **Management Mode:** Fully Managed Device Owner
- **Target Use Case:** Enterprise device anomaly detection for platforms like XSight and MobiControl
- **Privacy Compliance:** All data collection respects Android security policies and enterprise privacy requirements

---

## Taxonomy of Collectible Datapoints

### 1. Device and Build Information
- Device model, manufacturer, hardware info
- OS version, SDK level, security patch level
- Device identifiers (serial number, Android ID)
- Build fingerprint and hardware characteristics

### 2. Battery and Charging
- Battery level and percentage
- Charging status and power source
- Battery health and temperature
- Charging history and patterns

### 3. Network and Connectivity
- Active network type (WiFi, cellular, ethernet)
- WiFi signal strength (RSSI)
- Cellular signal strength and network type
- Connection state and transition events
- Data usage statistics (per app and aggregate)

### 4. Storage and Memory
- Available and total storage (internal/external)
- Storage usage by app (requires usage stats)
- Runtime memory statistics (free, total, max)

### 5. Application Inventory and State
- Installed applications list
- App versions and install sources
- App state (running, stopped, cached)
- App usage statistics (foreground time, launches)

### 6. Security Signals
- Device encryption status
- Screen lock type and password quality
- Security patch level
- Biometric availability
- Device admin status

### 7. Sensors and Hardware Status
- Bluetooth availability and state
- GPS/location services availability
- Camera availability
- NFC state
- Sensor availability (without continuous sampling)

### 8. Performance Metrics
- CPU usage (limited without privileged access)
- Process counts and running services
- ANR (Application Not Responding) events
- App crash events (Android 11+)

### 9. System Logs (Limited Access)
- Security logs (Device Owner only)
- Network logs (Device Owner only, Android 8+)
- Note: Full logcat access requires system app privileges

---

## Detailed Datapoint Specifications

### 1. Device and Build Information

#### 1.1 Device Model and Manufacturer

**API Name and Class:** `Build.MODEL`, `Build.MANUFACTURER`  
**Package:** `android.os.Build`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Static - retrieve once at initialization  
**Reliability Notes:** Highly reliable, consistent across all devices  
**Anomaly Detection Use Cases:** 
- Detect unauthorized device models in fleet
- Identify devices not matching corporate standards
- Track device lifecycle and model distribution

**Implementation:**
```java
String model = Build.MODEL;
String manufacturer = Build.MANUFACTURER;
String device = Build.DEVICE;
String hardware = Build.HARDWARE;
```

**References:** [Android Build Documentation](https://developer.android.com/reference/android/os/Build)

---

#### 1.2 Operating System Version

**API Name and Class:** `Build.VERSION.RELEASE`, `Build.VERSION.SDK_INT`  
**Package:** `android.os.Build`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Static - retrieve once, update on OS version changes  
**Reliability Notes:** Highly reliable  
**Anomaly Detection Use Cases:**
- Ensure devices meet minimum OS version requirements
- Detect devices running outdated or unsupported Android versions
- Track OS version distribution across fleet

**Implementation:**
```java
String osVersion = Build.VERSION.RELEASE;
int sdkVersion = Build.VERSION.SDK_INT;
String codename = Build.VERSION.CODENAME;
```

---

#### 1.3 Security Patch Level

**API Name and Class:** `Build.VERSION.SECURITY_PATCH`  
**Package:** `android.os.Build`  
**Minimum Android Version:** API level 23 (Android 6.0)  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Static - retrieve periodically to detect updates  
**Reliability Notes:** Returns null on devices without security patch level; reliable on supported devices  
**Anomaly Detection Use Cases:**
- Detect devices with outdated security patches
- Ensure compliance with security patch policies
- Identify devices vulnerable to known security issues

**Implementation:**
```java
String securityPatch = Build.VERSION.SECURITY_PATCH; // Format: "YYYY-MM-DD"
```

---

#### 1.4 Device Serial Number

**API Name and Class:** `Build.getSerial()`  
**Package:** `android.os.Build`  
**Minimum Android Version:** API level 9 (Android 2.3), but requires API 26+ for non-system apps  
**Required Permissions:** `READ_PHONE_STATE`  
**Enterprise Requirements:** Device Owner or Profile Owner (for non-system apps)  
**Data Cadence:** Static - retrieve once at initialization  
**Reliability Notes:** 
- Access restricted from Android 8.0 onwards for regular apps
- Device Owner apps can access without user approval
- May return "unknown" on some devices or if permission denied
**Anomaly Detection Use Cases:**
- Device identification and inventory tracking
- Detect device cloning or tampering
- Unique device identification in telemetry data

**Implementation:**
```java
// Requires READ_PHONE_STATE permission
String serial = Build.getSerial(); // May require Device Owner context
```

**References:** [Build.getSerial() Documentation](https://developer.android.com/reference/android/os/Build#getSerial())

---

#### 1.5 Android ID

**API Name and Class:** `Settings.Secure.ANDROID_ID`  
**Package:** `android.provider.Settings.Secure`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None (but may require READ_PRIVILEGED_PHONE_STATE for multi-user consistency)  
**Enterprise Requirements:** None  
**Data Cadence:** Static - retrieve once, cached per app signing key + user  
**Reliability Notes:** 
- Resets on factory reset
- Different per app signing key
- Different per user profile on multi-user devices
- Not reliable as a permanent device identifier
**Anomaly Detection Use Cases:**
- App-scoped device identification
- User profile identification on shared devices

**Implementation:**
```java
String androidId = Settings.Secure.getString(getContentResolver(), Settings.Secure.ANDROID_ID);
```

---

#### 1.6 Build Fingerprint

**API Name and Class:** `Build.FINGERPRINT`  
**Package:** `android.os.Build`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Static  
**Reliability Notes:** Unique identifier combining manufacturer, model, and build info  
**Anomaly Detection Use Cases:**
- Detect custom ROMs or modified firmware
- Identify specific build variants
- Track build distribution

**Implementation:**
```java
String fingerprint = Build.FINGERPRINT;
```

---

### 2. Battery and Charging

#### 2.1 Battery Level and Status

**API Name and Class:** `BatteryManager`, `Intent.ACTION_BATTERY_CHANGED`  
**Package:** `android.os.BatteryManager`  
**Minimum Android Version:** API level 1 (via BroadcastReceiver), API level 21 for BatteryManager methods  
**Required Permissions:** None for basic battery info  
**Enterprise Requirements:** None  
**Data Cadence:** Event-driven (BroadcastReceiver) or polling via BatteryManager  
**Reliability Notes:** Highly reliable; broadcast sent on every battery status change  
**Anomaly Detection Use Cases:**
- Detect rapid battery drain indicating hardware issues or resource-intensive apps
- Monitor charging patterns and identify charging anomalies
- Track battery health over time

**Implementation:**
```java
// Via BroadcastReceiver (recommended for real-time updates)
IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
Intent batteryStatus = context.registerReceiver(null, ifilter);
int level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
int scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
float batteryPct = level * 100 / (float)scale;

int status = batteryStatus.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
boolean isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                     status == BatteryManager.BATTERY_STATUS_FULL;

// Via BatteryManager (API 21+)
BatteryManager bm = (BatteryManager) context.getSystemService(Context.BATTERY_SERVICE);
int batteryLevel = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
```

**References:** [BatteryManager Documentation](https://developer.android.com/reference/android/os/BatteryManager)

---

#### 2.2 Battery Health

**API Name and Class:** `BatteryManager.EXTRA_HEALTH`  
**Package:** `android.os.BatteryManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Event-driven via ACTION_BATTERY_CHANGED broadcast  
**Reliability Notes:** Provides general health status (GOOD, OVERHEAT, DEAD, etc.)  
**Anomaly Detection Use Cases:**
- Identify failing batteries
- Detect overheating conditions
- Monitor battery health degradation over time

**Implementation:**
```java
int health = batteryStatus.getIntExtra(BatteryManager.EXTRA_HEALTH, -1);
// Values: BATTERY_HEALTH_GOOD, BATTERY_HEALTH_OVERHEAT, 
//         BATTERY_HEALTH_DEAD, BATTERY_HEALTH_OVER_VOLTAGE, etc.
```

---

#### 2.3 Battery Technology and Temperature

**API Name and Class:** `BatteryManager.EXTRA_TECHNOLOGY`, `BatteryManager.EXTRA_TEMPERATURE`  
**Package:** `android.os.BatteryManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Event-driven via ACTION_BATTERY_CHANGED  
**Reliability Notes:** Temperature in tenths of a degree Celsius  
**Anomaly Detection Use Cases:**
- Detect overheating conditions
- Monitor charging temperature patterns

**Implementation:**
```java
String technology = batteryStatus.getStringExtra(BatteryManager.EXTRA_TECHNOLOGY);
int temperature = batteryStatus.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1); // in 0.1°C
```

---

#### 2.4 Charging Power Source

**API Name and Class:** `BatteryManager.EXTRA_PLUGGED`  
**Package:** `android.os.BatteryManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Event-driven via ACTION_BATTERY_CHANGED  
**Reliability Notes:** Indicates USB, AC, or wireless charging  
**Anomaly Detection Use Cases:**
- Monitor charging patterns
- Detect unauthorized charging sources
- Track charging efficiency

**Implementation:**
```java
int plugged = batteryStatus.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1);
boolean usbCharge = plugged == BatteryManager.BATTERY_PLUGGED_USB;
boolean acCharge = plugged == BatteryManager.BATTERY_PLUGGED_AC;
boolean wirelessCharge = plugged == BatteryManager.BATTERY_PLUGGED_WIRELESS;
```

---

### 3. Network and Connectivity

#### 3.1 Network Connectivity Status

**API Name and Class:** `ConnectivityManager.getActiveNetworkInfo()`, `NetworkCallback`  
**Package:** `android.net.ConnectivityManager`  
**Minimum Android Version:** API level 1 (deprecated methods), API level 21+ (NetworkCallback)  
**Required Permissions:** `ACCESS_NETWORK_STATE`  
**Enterprise Requirements:** None  
**Data Cadence:** Event-driven via NetworkCallback or polling  
**Reliability Notes:** 
- `getActiveNetworkInfo()` deprecated in API 28+, use `getActiveNetwork()` and `NetworkCapabilities`
- NetworkCallback provides real-time updates
**Anomaly Detection Use Cases:**
- Detect unexpected network disconnections
- Monitor network connectivity patterns
- Identify devices frequently losing connectivity

**Implementation:**
```java
ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);

// API 23+ (recommended)
Network network = cm.getActiveNetwork();
NetworkCapabilities caps = cm.getNetworkCapabilities(network);
boolean hasInternet = caps != null && caps.hasCapability(
    NetworkCapabilities.NET_CAPABILITY_INTERNET);

// NetworkCallback for real-time updates
NetworkRequest request = new NetworkRequest.Builder()
    .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    .build();
cm.registerNetworkCallback(request, networkCallback);
```

**References:** [ConnectivityManager Documentation](https://developer.android.com/reference/android/net/ConnectivityManager)

---

#### 3.2 Network Type (WiFi, Cellular, Ethernet)

**API Name and Class:** `NetworkCapabilities`, `ConnectivityManager`  
**Package:** `android.net`  
**Minimum Android Version:** API level 21  
**Required Permissions:** `ACCESS_NETWORK_STATE`  
**Enterprise Requirements:** None  
**Data Cadence:** Event-driven via NetworkCallback or polling  
**Reliability Notes:** More reliable than deprecated `getNetworkType()`  
**Anomaly Detection Use Cases:**
- Monitor network type transitions
- Detect unauthorized network types (e.g., cellular when WiFi required)
- Track network usage patterns

**Implementation:**
```java
NetworkCapabilities caps = cm.getNetworkCapabilities(network);
boolean hasWifi = caps != null && caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI);
boolean hasCellular = caps != null && caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR);
boolean hasEthernet = caps != null && caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET);
```

---

#### 3.3 WiFi Signal Strength (RSSI)

**API Name and Class:** `WifiInfo.getRssi()`, `WifiManager`  
**Package:** `android.net.wifi`  
**Minimum Android Version:** API level 1  
**Required Permissions:** `ACCESS_WIFI_STATE`  
**Enterprise Requirements:** None  
**Data Cadence:** Polling or event-driven via WifiManager  
**Reliability Notes:** 
- RSSI values in dBm (typically -100 to 0, where higher is better)
- Only valid when WiFi is connected
- May require location permission on some Android versions (check at runtime)
**Anomaly Detection Use Cases:**
- Detect weak WiFi signals indicating connectivity issues
- Monitor WiFi signal stability
- Identify devices with poor network positioning

**Implementation:**
```java
WifiManager wifiManager = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
WifiInfo wifiInfo = wifiManager.getConnectionInfo();
int rssi = wifiInfo.getRssi(); // Signal strength in dBm
String ssid = wifiInfo.getSSID(); // May return "<unknown ssid>" on Android 8+ without location permission
int linkSpeed = wifiInfo.getLinkSpeed(); // in Mbps
```

**Note:** On Android 8.0+, accessing SSID/BSSID may require location permissions due to privacy restrictions. Device Owner apps may have exemptions.

**References:** [WifiManager Documentation](https://developer.android.com/reference/android/net/wifi/WifiManager)

---

#### 3.4 Cellular Network Type and Signal Strength

**API Name and Class:** `TelephonyManager.getNetworkType()`, `TelephonyManager.getCellLocation()`  
**Package:** `android.telephony.TelephonyManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** `READ_PHONE_STATE`  
**Enterprise Requirements:** Device Owner recommended for reliable access  
**Data Cadence:** Polling  
**Reliability Notes:** 
- `getNetworkType()` deprecated in API 28+, use `getDataNetworkType()` or `getVoiceNetworkType()`
- Cell location requires additional permissions and may be restricted
- Signal strength requires listening to `PhoneStateListener`
**Anomaly Detection Use Cases:**
- Monitor cellular network quality
- Detect network type downgrades (e.g., 5G to 4G)
- Track signal strength patterns

**Implementation:**
```java
TelephonyManager tm = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);

// Network type (deprecated in API 28+)
int networkType = tm.getNetworkType(); // NETWORK_TYPE_LTE, NETWORK_TYPE_5G, etc.

// API 26+ (recommended)
int dataNetworkType = tm.getDataNetworkType();
int voiceNetworkType = tm.getVoiceNetworkType();

// Signal strength via PhoneStateListener
PhoneStateListener listener = new PhoneStateListener() {
    @Override
    public void onSignalStrengthsChanged(SignalStrength signalStrength) {
        int level = signalStrength.getLevel(); // 0-4
        int dbm = signalStrength.getDbm(); // Signal strength in dBm
    }
};
tm.listen(listener, PhoneStateListener.LISTEN_SIGNAL_STRENGTHS);
```

**References:** [TelephonyManager Documentation](https://developer.android.com/reference/android/telephony/TelephonyManager)

---

#### 3.5 Network Data Usage Statistics

**API Name and Class:** `NetworkStatsManager.querySummaryForDevice()`  
**Package:** `android.app.usage.NetworkStatsManager`  
**Minimum Android Version:** API level 23 (Android 6.0)  
**Required Permissions:** `PACKAGE_USAGE_STATS` (must be granted by user via Settings)  
**Enterprise Requirements:** Device Owner can grant usage stats access programmatically  
**Data Cadence:** Polling - query aggregated stats over time intervals  
**Reliability Notes:** 
- Requires user to grant PACKAGE_USAGE_STATS in Settings
- Device Owner can request usage stats access via `DevicePolicyManager.setPermissionGrantState()`
- Data is aggregated and may have delays
- May not include all network interfaces on all devices
**Anomaly Detection Use Cases:**
- Detect unusual data consumption patterns
- Identify apps consuming excessive data
- Monitor data usage trends for capacity planning
- Detect potential data leaks or unauthorized data transfers

**Implementation:**
```java
NetworkStatsManager nsm = (NetworkStatsManager) context.getSystemService(Context.NETWORK_STATS_SERVICE);

// Query device-level stats
NetworkStats.Bucket bucket = nsm.querySummaryForDevice(
    ConnectivityManager.TYPE_WIFI,
    null, // subscriberId - null for WiFi
    startTime,
    endTime
);
long rxBytes = bucket.getRxBytes();
long txBytes = bucket.getTxBytes();

// Query per-UID stats (requires PACKAGE_USAGE_STATS)
NetworkStats stats = nsm.querySummary(
    ConnectivityManager.TYPE_WIFI,
    null,
    startTime,
    endTime
);
```

**Device Owner Enhancement:** Device Owner apps can use `DevicePolicyManager.setPermissionGrantState()` to grant PACKAGE_USAGE_STATS to themselves.

**References:** [NetworkStatsManager Documentation](https://developer.android.com/reference/android/app/usage/NetworkStatsManager)

---

### 4. Storage and Memory

#### 4.1 Available and Total Storage

**API Name and Class:** `StatFs.getAvailableBytes()`, `StatFs.getTotalBytes()`  
**Package:** `android.os.StatFs`  
**Minimum Android Version:** API level 1 (methods deprecated), API level 18+ for getAvailableBytes/getTotalBytes  
**Required Permissions:** None for internal storage; `READ_EXTERNAL_STORAGE` for external storage  
**Enterprise Requirements:** None  
**Data Cadence:** Polling  
**Reliability Notes:** Accurate for internal storage; external storage availability varies  
**Anomaly Detection Use Cases:**
- Detect low storage conditions
- Monitor storage depletion trends
- Identify sudden storage consumption spikes
- Alert on storage threshold breaches

**Implementation:**
```java
// Internal storage
StatFs stat = new StatFs(Environment.getDataDirectory().getPath());
long totalBytes = stat.getTotalBytes();
long availableBytes = stat.getAvailableBytes();
long freeBytes = stat.getFreeBytes(); // May differ from available on some systems

// External storage (requires READ_EXTERNAL_STORAGE on API 19+)
File externalDir = Environment.getExternalStorageDirectory();
if (externalDir != null && externalDir.exists()) {
    StatFs externalStat = new StatFs(externalDir.getPath());
    long externalTotal = externalStat.getTotalBytes();
    long externalAvailable = externalStat.getAvailableBytes();
}
```

**References:** [StatFs Documentation](https://developer.android.com/reference/android/os/StatFs)

---

#### 4.2 Storage Usage by Application

**API Name and Class:** `StorageStatsManager.queryStatsForPackage()`, `StorageStatsManager.queryStatsForUid()`  
**Package:** `android.app.usage.StorageStatsManager`  
**Minimum Android Version:** API level 26 (Android 8.0)  
**Required Permissions:** `PACKAGE_USAGE_STATS`  
**Enterprise Requirements:** Device Owner can grant usage stats access  
**Data Cadence:** Polling  
**Reliability Notes:** Requires PACKAGE_USAGE_STATS permission  
**Anomaly Detection Use Cases:**
- Identify apps consuming excessive storage
- Monitor storage usage trends per application
- Detect storage leaks or cache buildup

**Implementation:**
```java
StorageStatsManager ssm = (StorageStatsManager) context.getSystemService(Context.STORAGE_STATS_SERVICE);
ApplicationInfo ai = context.getPackageManager().getApplicationInfo(packageName, 0);

try {
    StorageStats stats = ssm.queryStatsForPackage(
        StorageManager.UUID_DEFAULT,
        packageName,
        UserHandle.getUserHandleForUid(ai.uid)
    );
    long appBytes = stats.getAppBytes();
    long dataBytes = stats.getDataBytes();
    long cacheBytes = stats.getCacheBytes();
} catch (PackageManager.NameNotFoundException | IOException e) {
    // Handle error
}
```

**References:** [StorageStatsManager Documentation](https://developer.android.com/reference/android/app/usage/StorageStatsManager)

---

#### 4.3 Runtime Memory Statistics

**API Name and Class:** `Runtime.getRuntime()`, `ActivityManager.MemoryInfo`  
**Package:** `java.lang.Runtime`, `android.app.ActivityManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None for Runtime; none for basic MemoryInfo  
**Enterprise Requirements:** None  
**Data Cadence:** Polling  
**Reliability Notes:** 
- Runtime memory is app-specific
- ActivityManager.MemoryInfo provides system-wide memory info
- May not be perfectly accurate due to garbage collection
**Anomaly Detection Use Cases:**
- Monitor memory pressure on device
- Detect memory leaks (trend analysis over time)
- Identify devices with insufficient available memory

**Implementation:**
```java
// App-level memory
Runtime runtime = Runtime.getRuntime();
long maxMemory = runtime.maxMemory();
long totalMemory = runtime.totalMemory();
long freeMemory = runtime.freeMemory();
long usedMemory = totalMemory - freeMemory;

// System-wide memory
ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
ActivityManager.MemoryInfo memInfo = new ActivityManager.MemoryInfo();
am.getMemoryInfo(memInfo);
long totalMem = memInfo.totalMem;
long availableMem = memInfo.availMem;
long threshold = memInfo.threshold;
boolean lowMemory = memInfo.lowMemory;
```

**References:** [ActivityManager Documentation](https://developer.android.com/reference/android/app/ActivityManager)

---

### 5. Application Inventory and State

#### 5.1 Installed Applications List

**API Name and Class:** `PackageManager.getInstalledApplications()`, `PackageManager.getInstalledPackages()`  
**Package:** `android.content.pm.PackageManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None (basic list), `QUERY_ALL_PACKAGES` required on Android 11+ for complete list  
**Enterprise Requirements:** Device Owner apps can query all packages on Android 11+  
**Data Cadence:** Polling or event-driven via PackageMonitor  
**Reliability Notes:** 
- On Android 11+, apps cannot see all installed packages without QUERY_ALL_PACKAGES
- Device Owner apps are exempt from this restriction
- Package install/uninstall events can be monitored via BroadcastReceiver
**Anomaly Detection Use Cases:**
- Detect unauthorized app installations
- Track app inventory changes
- Ensure compliance with app whitelist/blacklist policies
- Identify apps not in corporate app catalog

**Implementation:**
```java
PackageManager pm = context.getPackageManager();

// Get all installed packages
List<ApplicationInfo> apps = pm.getInstalledApplications(PackageManager.GET_META_DATA);
for (ApplicationInfo app : apps) {
    String packageName = app.packageName;
    String appName = pm.getApplicationLabel(app).toString();
    boolean isSystemApp = (app.flags & ApplicationInfo.FLAG_SYSTEM) != 0;
}

// Monitor package changes
BroadcastReceiver packageReceiver = new BroadcastReceiver() {
    @Override
    public void onReceive(Context context, Intent intent) {
        String packageName = intent.getData().getSchemeSpecificPart();
        if (Intent.ACTION_PACKAGE_ADDED.equals(intent.getAction())) {
            // Package installed
        } else if (Intent.ACTION_PACKAGE_REMOVED.equals(intent.getAction())) {
            // Package removed
        }
    }
};
IntentFilter filter = new IntentFilter(Intent.ACTION_PACKAGE_ADDED);
filter.addAction(Intent.ACTION_PACKAGE_REMOVED);
filter.addDataScheme("package");
context.registerReceiver(packageReceiver, filter);
```

**Note:** On Android 11+, Device Owner apps can query all packages without QUERY_ALL_PACKAGES permission.

**References:** [PackageManager Documentation](https://developer.android.com/reference/android/content/pm/PackageManager)

---

#### 5.2 Application Version and Install Source

**API Name and Class:** `PackageManager.getPackageInfo()`  
**Package:** `android.content.pm.PackageManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Polling  
**Reliability Notes:** Reliable for installed packages  
**Anomaly Detection Use Cases:**
- Detect outdated app versions
- Identify apps installed from unauthorized sources
- Track app version distribution

**Implementation:**
```java
try {
    PackageInfo pkgInfo = pm.getPackageInfo(packageName, 0);
    String versionName = pkgInfo.versionName;
    int versionCode = pkgInfo.versionCode; // or long versionCodeLong on API 28+
    long installTime = pkgInfo.firstInstallTime;
    long updateTime = pkgInfo.lastUpdateTime;
    
    // Install source (requires GET_META_DATA flag and appropriate permissions)
    PackageInfo pkgInfoFull = pm.getPackageInfo(packageName, PackageManager.GET_META_DATA);
    String installerPackage = pm.getInstallerPackageName(packageName);
} catch (PackageManager.NameNotFoundException e) {
    // Package not found
}
```

---

#### 5.3 Application Usage Statistics

**API Name and Class:** `UsageStatsManager.queryUsageStats()`  
**Package:** `android.app.usage.UsageStatsManager`  
**Minimum Android Version:** API level 21 (Android 5.0)  
**Required Permissions:** `PACKAGE_USAGE_STATS` (must be granted by user)  
**Enterprise Requirements:** Device Owner can grant this permission programmatically  
**Data Cadence:** Polling - query aggregated stats over time intervals  
**Reliability Notes:** 
- Requires user grant of PACKAGE_USAGE_STATS in Settings
- Device Owner can use `DevicePolicyManager.setPermissionGrantState()` to grant access
- Data granularity: minimum 1 minute, aggregated
- May not include all apps (system may filter)
**Anomaly Detection Use Cases:**
- Detect unauthorized app usage
- Monitor app usage patterns and productivity
- Identify apps consuming excessive resources
- Track foreground app time per application
- Detect unusual app launch patterns

**Implementation:**
```java
UsageStatsManager usm = (UsageStatsManager) context.getSystemService(Context.USAGE_STATS_SERVICE);

// Query usage stats for a time interval
long endTime = System.currentTimeMillis();
long startTime = endTime - (24 * 60 * 60 * 1000); // Last 24 hours

List<UsageStats> stats = usm.queryUsageStats(
    UsageStatsManager.INTERVAL_DAILY,
    startTime,
    endTime
);

for (UsageStats stat : stats) {
    String packageName = stat.getPackageName();
    long totalTimeInForeground = stat.getTotalTimeInForeground(); // milliseconds
    long lastTimeUsed = stat.getLastTimeUsed();
    long lastTimeVisible = stat.getLastTimeVisible();
    int launchCount = stat.getLaunchCount();
}

// Query events (requires API 29+)
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
    UsageEvents events = usm.queryEvents(startTime, endTime);
    UsageEvents.Event event = new UsageEvents.Event();
    while (events.hasNextEvent()) {
        events.getNextEvent(event);
        String packageName = event.getPackageName();
        int eventType = event.getEventType(); // MOVE_TO_FOREGROUND, MOVE_TO_BACKGROUND, etc.
        long timeStamp = event.getTimeStamp();
    }
}
```

**Device Owner Enhancement:** Device Owner apps can grant PACKAGE_USAGE_STATS to themselves:
```java
DevicePolicyManager dpm = (DevicePolicyManager) context.getSystemService(Context.DEVICE_POLICY_SERVICE);
ComponentName admin = new ComponentName(context, DeviceAdminReceiver.class);
dpm.setPermissionGrantState(admin, packageName, Manifest.permission.PACKAGE_USAGE_STATS, 
    DevicePolicyManager.PERMISSION_GRANT_STATE_GRANTED);
```

**References:** [UsageStatsManager Documentation](https://developer.android.com/reference/android/app/usage/UsageStatsManager)

---

### 6. Security Signals

#### 6.1 Device Encryption Status

**API Name and Class:** `DevicePolicyManager.getStorageEncryptionStatus()`  
**Package:** `android.app.admin.DevicePolicyManager`  
**Minimum Android Version:** API level 11 (Android 3.0)  
**Required Permissions:** `BIND_DEVICE_ADMIN` (must be Device Owner or Profile Owner)  
**Enterprise Requirements:** Device Owner or Profile Owner  
**Data Cadence:** Polling  
**Reliability Notes:** Reliable when app is Device Owner  
**Anomaly Detection Use Cases:**
- Ensure devices are encrypted (compliance requirement)
- Detect unauthorized encryption status changes
- Monitor encryption status across fleet

**Implementation:**
```java
DevicePolicyManager dpm = (DevicePolicyManager) context.getSystemService(Context.DEVICE_POLICY_SERVICE);
ComponentName admin = new ComponentName(context, DeviceAdminReceiver.class);

int encryptionStatus = dpm.getStorageEncryptionStatus(admin);
// Values: ENCRYPTION_STATUS_UNSUPPORTED, ENCRYPTION_STATUS_INACTIVE,
//         ENCRYPTION_STATUS_ACTIVATING, ENCRYPTION_STATUS_ACTIVE,
//         ENCRYPTION_STATUS_ACTIVE_DEFAULT_KEY, ENCRYPTION_STATUS_ACTIVE_PER_USER
```

**References:** [DevicePolicyManager Documentation](https://developer.android.com/reference/android/app/admin/DevicePolicyManager)

---

#### 6.2 Screen Lock and Password Quality

**API Name and Class:** `DevicePolicyManager.getPasswordQuality()`, `KeyguardManager.isDeviceSecure()`  
**Package:** `android.app.admin.DevicePolicyManager`, `android.app.KeyguardManager`  
**Minimum Android Version:** API level 8 (DevicePolicyManager), API level 23 (KeyguardManager.isDeviceSecure)  
**Required Permissions:** `BIND_DEVICE_ADMIN` for DevicePolicyManager; none for KeyguardManager  
**Enterprise Requirements:** Device Owner for DevicePolicyManager methods; none for KeyguardManager  
**Data Cadence:** Polling  
**Reliability Notes:** Reliable when Device Owner  
**Anomaly Detection Use Cases:**
- Ensure compliance with password policy requirements
- Detect weak passwords or disabled screen locks
- Monitor password quality across fleet

**Implementation:**
```java
// Device Owner approach
int passwordQuality = dpm.getPasswordQuality(admin);
// Values: PASSWORD_QUALITY_UNSPECIFIED, PASSWORD_QUALITY_SOMETHING,
//         PASSWORD_QUALITY_BIOMETRIC_WEAK, PASSWORD_QUALITY_NUMERIC,
//         PASSWORD_QUALITY_ALPHABETIC, PASSWORD_QUALITY_ALPHANUMERIC,
//         PASSWORD_QUALITY_COMPLEX

int minLength = dpm.getPasswordMinimumLength(admin);
boolean hasPassword = dpm.isActivePasswordSufficient(admin);

// Non-admin approach (API 23+)
KeyguardManager km = (KeyguardManager) context.getSystemService(Context.KEYGUARD_SERVICE);
boolean isDeviceSecure = km.isDeviceSecure();
boolean isKeyguardSecure = km.isKeyguardSecure();
```

---

#### 6.3 Security Patch Level

**API Name and Class:** `Build.VERSION.SECURITY_PATCH`  
**Package:** `android.os.Build`  
**Minimum Android Version:** API level 23 (Android 6.0)  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Static - check periodically  
**Reliability Notes:** Returns null on devices without security patch level  
**Anomaly Detection Use Cases:**
- Ensure devices have recent security patches
- Detect devices vulnerable to known security issues
- Track security patch compliance across fleet

**Implementation:**
```java
String securityPatch = Build.VERSION.SECURITY_PATCH; // Format: "YYYY-MM-DD"
// Compare with current date to determine patch age
```

---

#### 6.4 Biometric Availability

**API Name and Class:** `BiometricManager.canAuthenticate()`  
**Package:** `android.hardware.biometrics.BiometricManager`  
**Minimum Android Version:** API level 29 (Android 10)  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Polling  
**Reliability Notes:** Indicates if biometric hardware is available and enrolled  
**Anomaly Detection Use Cases:**
- Verify biometric authentication availability
- Track biometric enrollment status

**Implementation:**
```java
BiometricManager bm = (BiometricManager) context.getSystemService(Context.BIOMETRIC_SERVICE);
int canAuthenticate = bm.canAuthenticate();
// Values: BIOMETRIC_SUCCESS, BIOMETRIC_ERROR_NO_HARDWARE,
//         BIOMETRIC_ERROR_HW_UNAVAILABLE, BIOMETRIC_ERROR_NONE_ENROLLED
```

---

### 7. Sensors and Hardware Status

#### 7.1 Bluetooth Availability and State

**API Name and Class:** `BluetoothAdapter.getDefaultAdapter()`, `BluetoothAdapter.getState()`  
**Package:** `android.bluetooth.BluetoothAdapter`  
**Minimum Android Version:** API level 1  
**Required Permissions:** `BLUETOOTH` (for basic state), `BLUETOOTH_ADMIN` (for state changes)  
**Enterprise Requirements:** None  
**Data Cadence:** Polling or event-driven via BroadcastReceiver  
**Reliability Notes:** Reliable for state queries  
**Anomaly Detection Use Cases:**
- Monitor Bluetooth state (enabled/disabled)
- Detect unauthorized Bluetooth usage
- Track Bluetooth availability

**Implementation:**
```java
BluetoothAdapter adapter = BluetoothAdapter.getDefaultAdapter();
if (adapter != null) {
    boolean isEnabled = adapter.isEnabled();
    int state = adapter.getState(); // STATE_OFF, STATE_ON, STATE_TURNING_ON, STATE_TURNING_OFF
    String address = adapter.getAddress(); // May be restricted on Android 6+ without location permission
    String name = adapter.getName();
}

// Monitor state changes
BroadcastReceiver btReceiver = new BroadcastReceiver() {
    @Override
    public void onReceive(Context context, Intent intent) {
        int state = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, -1);
        // Handle state change
    }
};
IntentFilter filter = new IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED);
context.registerReceiver(btReceiver, filter);
```

**Note:** MAC address access may require location permissions on Android 6+.

---

#### 7.2 GPS/Location Services Availability

**API Name and Class:** `LocationManager.isProviderEnabled()`  
**Package:** `android.location.LocationManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None for checking provider availability  
**Enterprise Requirements:** None  
**Data Cadence:** Polling or event-driven via Settings change listener  
**Reliability Notes:** Reliable for checking if location services are enabled  
**Anomaly Detection Use Cases:**
- Monitor location services state (not actual location)
- Detect disabled location services
- Track location service availability

**Implementation:**
```java
LocationManager lm = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
boolean gpsEnabled = lm.isProviderEnabled(LocationManager.GPS_PROVIDER);
boolean networkEnabled = lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER);
boolean passiveEnabled = lm.isProviderEnabled(LocationManager.PASSIVE_PROVIDER);
```

**Note:** This only checks if providers are enabled, not actual location data. Location data collection requires additional permissions and user consent.

---

#### 7.3 Camera Availability

**API Name and Class:** `PackageManager.hasSystemFeature()`  
**Package:** `android.content.pm.PackageManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None  
**Enterprise Requirements:** None  
**Data Cadence:** Static - check once  
**Reliability Notes:** Reliable  
**Anomaly Detection Use Cases:**
- Verify camera hardware availability
- Track device capabilities

**Implementation:**
```java
PackageManager pm = context.getPackageManager();
boolean hasCamera = pm.hasSystemFeature(PackageManager.FEATURE_CAMERA);
boolean hasFrontCamera = pm.hasSystemFeature(PackageManager.FEATURE_CAMERA_FRONT);
```

---

#### 7.4 NFC State

**API Name and Class:** `NfcAdapter.getDefaultAdapter()`, `NfcAdapter.isEnabled()`  
**Package:** `android.nfc.NfcAdapter`  
**Minimum Android Version:** API level 9 (Android 2.3)  
**Required Permissions:** `NFC`  
**Enterprise Requirements:** None  
**Data Cadence:** Polling or event-driven  
**Reliability Notes:** Returns null if NFC not available  
**Anomaly Detection Use Cases:**
- Monitor NFC state
- Track NFC availability

**Implementation:**
```java
NfcAdapter nfcAdapter = NfcAdapter.getDefaultAdapter(context);
if (nfcAdapter != null) {
    boolean isEnabled = nfcAdapter.isEnabled();
}
```

---

### 8. Performance Metrics

#### 8.1 Running Processes and Services

**API Name and Class:** `ActivityManager.getRunningAppProcesses()`, `ActivityManager.getRunningServices()`  
**Package:** `android.app.ActivityManager`  
**Minimum Android Version:** API level 1  
**Required Permissions:** None (but limited on newer Android versions)  
**Enterprise Requirements:** None, but results limited on Android 8+  
**Data Cadence:** Polling  
**Reliability Notes:** 
- `getRunningAppProcesses()` returns only app's own processes on Android 8+
- `getRunningServices()` has similar limitations
- Cannot reliably enumerate all system processes without system app privileges
**Anomaly Detection Use Cases:**
- Monitor app's own process state
- Limited anomaly detection due to restrictions

**Implementation:**
```java
ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);

// Running processes (limited on Android 8+)
List<ActivityManager.RunningAppProcessInfo> processes = am.getRunningAppProcesses();
for (ActivityManager.RunningAppProcessInfo proc : processes) {
    String processName = proc.processName;
    int pid = proc.pid;
    int importance = proc.importance; // FOREGROUND, VISIBLE, SERVICE, etc.
}

// Running services (also limited)
List<ActivityManager.RunningServiceInfo> services = am.getRunningServices(Integer.MAX_VALUE);
```

**Limitations:** On Android 8.0+, these methods only return information about the calling app's own processes/services. For comprehensive process monitoring, system app privileges are required.

---

#### 8.2 Application Exit Reasons (ANR and Crashes)

**API Name and Class:** `ActivityManager.getHistoricalProcessExitReasons()`, `ApplicationExitInfo`  
**Package:** `android.app.ActivityManager`, `android.app.ApplicationExitInfo`  
**Minimum Android Version:** API level 30 (Android 11)  
**Required Permissions:** None for own app's exit reasons  
**Enterprise Requirements:** None  
**Data Cadence:** Polling - query historical exit reasons  
**Reliability Notes:** Only returns exit reasons for the calling app (or apps with same UID in debug builds)  
**Anomaly Detection Use Cases:**
- Monitor app crashes and ANRs for the telemetry agent itself
- Track stability metrics
- Note: Cannot reliably monitor other apps' crashes without system privileges

**Implementation:**
```java
ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);

if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
    List<ApplicationExitInfo> exitReasons = am.getHistoricalProcessExitReasons(
        context.getPackageName(),
        0, // pid - 0 for all PIDs
        5  // maxNum - limit results
    );
    
    for (ApplicationExitInfo info : exitReasons) {
        int reason = info.getReason(); // REASON_ANR, REASON_CRASH, REASON_EXIT_SELF, etc.
        int status = info.getStatus();
        long timestamp = info.getTimestamp();
        String description = info.getDescription();
        int pid = info.getPid();
        
        // Get associated process state if available
        ApplicationExitInfo.ProcessStateSnapshot snapshot = info.getProcessStateSnapshot();
        if (snapshot != null) {
            int importance = snapshot.getImportance();
            int oomScore = snapshot.getOomScore();
        }
    }
}
```

**Limitations:** This API only provides exit reasons for the calling app itself. To monitor other apps' crashes, system app privileges or crash reporting SDKs (Firebase Crashlytics, etc.) would be required.

**References:** [ApplicationExitInfo Documentation](https://developer.android.com/reference/android/app/ApplicationExitInfo)

---

### 9. System Logs (Device Owner Only)

#### 9.1 Security Logs

**API Name and Class:** `DevicePolicyManager.retrieveSecurityLogs()`  
**Package:** `android.app.admin.DevicePolicyManager`  
**Minimum Android Version:** API level 24 (Android 7.0)  
**Required Permissions:** Device Owner role  
**Enterprise Requirements:** Device Owner (not Profile Owner)  
**Data Cadence:** Event-driven - retrieve logs when available  
**Reliability Notes:** 
- Requires Device Owner role
- Security logging must be enabled via `setSecurityLoggingEnabled()`
- Logs are batched and may have delays
- Some OEMs may restrict or modify security logging behavior
**Anomaly Detection Use Cases:**
- Monitor security-relevant events (screen unlock attempts, policy changes, etc.)
- Detect unauthorized access attempts
- Track security policy violations
- Forensic analysis of security incidents

**Implementation:**
```java
DevicePolicyManager dpm = (DevicePolicyManager) context.getSystemService(Context.DEVICE_POLICY_SERVICE);
ComponentName admin = new ComponentName(context, DeviceAdminReceiver.class);

// Enable security logging
dpm.setSecurityLoggingEnabled(admin, true);

// Check if logs are available
boolean areLogsReady = dpm.isSecurityLoggingEnabled(admin);

// Retrieve security logs
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && areLogsReady) {
    List<SecurityEvent> events = dpm.retrieveSecurityLogs(admin);
    
    for (SecurityEvent event : events) {
        int tag = event.getTag();
        long timeNanos = event.getTimeNanos();
        ByteBuffer data = event.getData();
        
        // Parse event based on tag
        // Common tags: TAG_ADB_SHELL_INTERACTIVE, TAG_ADB_SHELL_CMD,
        //              TAG_APP_PROCESS_START, TAG_CERT_AUTHORITY_INSTALLED,
        //              TAG_CERT_AUTHORITY_REMOVED, TAG_KEY_DESTRUCTION,
        //              TAG_KEY_GENERATION, TAG_KEY_IMPORT, TAG_KEY_INTEGRITY_VIOLATION,
        //              TAG_LOGGING_STARTED, TAG_LOGGING_STOPPED, TAG_LOG_BUFFER_SIZE_CRITICAL,
        //              TAG_MAX_PASSWORD_ATTEMPTS, TAG_MAX_SCREEN_LOCK_TIMEOUT,
        //              TAG_OS_STARTUP, TAG_OS_SHUTDOWN, TAG_PASSWORD_COMPLEXITY_SET,
        //              TAG_PASSWORD_EXPIRATION_SET, TAG_PASSWORD_HISTORY_LENGTH_SET,
        //              TAG_REMOTE_LOCK, TAG_SYNC_RECV_FILE, TAG_SYNC_SEND_FILE,
        //              TAG_USER_RESTRICTION_ADDED, TAG_USER_RESTRICTION_REMOVED,
        //              TAG_WIPE_FAILURE, etc.
    }
}
```

**Device Owner Requirement:** This API is **exclusively** available to Device Owner apps, not Profile Owner apps.

**References:** [DevicePolicyManager Security Logging](https://developer.android.com/work/dpc/logging#security_logging)

---

#### 9.2 Network Logs

**API Name and Class:** `DevicePolicyManager.retrieveNetworkLogs()`  
**Package:** `android.app.admin.DevicePolicyManager`  
**Minimum Android Version:** API level 26 (Android 8.0)  
**Required Permissions:** Device Owner role  
**Enterprise Requirements:** Device Owner (not Profile Owner)  
**Data Cadence:** Event-driven - retrieve logs periodically (typically hourly)  
**Reliability Notes:** 
- Requires Device Owner role
- Network logging must be enabled via `setNetworkLoggingEnabled()`
- Logs are batched and retrieved periodically (typically every hour)
- Contains DNS lookup events and network connections
- Some OEMs may restrict network logging behavior
**Anomaly Detection Use Cases:**
- Monitor network connections and DNS lookups
- Detect unauthorized network access
- Track network usage patterns
- Identify potential security threats or data exfiltration attempts

**Implementation:**
```java
// Enable network logging
dpm.setNetworkLoggingEnabled(admin, true);

// Check if logs are available
boolean areNetworkLogsReady = dpm.isNetworkLoggingEnabled(admin);

// Retrieve network logs (API 26+)
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && areNetworkLogsReady) {
    List<NetworkEvent> events = dpm.retrieveNetworkLogs(admin);
    
    for (NetworkEvent event : events) {
        String packageName = event.getPackageName();
        long timestamp = event.getTimestamp();
        
        if (event instanceof DnsEvent) {
            DnsEvent dnsEvent = (DnsEvent) event;
            String hostname = dnsEvent.getHostname();
            List<InetAddress> ipAddresses = dnsEvent.getIpAddresses();
            int totalResolvedAddressCount = dnsEvent.getTotalResolvedAddressCount();
        } else if (event instanceof ConnectEvent) {
            ConnectEvent connectEvent = (ConnectEvent) event;
            InetAddress inetAddress = connectEvent.getInetAddress();
            int port = connectEvent.getPort();
            String ipAddress = inetAddress.getHostAddress();
        }
    }
}
```

**Device Owner Requirement:** This API is **exclusively** available to Device Owner apps.

**References:** [DevicePolicyManager Network Logging](https://developer.android.com/work/dpc/logging#network_logging)

---

#### 9.3 Logcat Access (System Apps Only)

**API Name and Class:** `Log`, `Runtime.exec("logcat")`  
**Package:** `android.util.Log`, `java.lang.Runtime`  
**Minimum Android Version:** API level 1  
**Required Permissions:** `READ_LOGS` (only available to system apps signed with platform key)  
**Enterprise Requirements:** System app or rooted device (not practical for enterprise apps)  
**Data Cadence:** N/A - not accessible to regular apps  
**Reliability Notes:** **Not available to third-party apps or Device Owner apps**  
**Anomaly Detection Use Cases:** N/A - not accessible

**Why Not Available:**
- `READ_LOGS` permission is a signature-level permission
- Only granted to apps signed with the platform key (system apps)
- Cannot be granted to Device Owner apps or any third-party app
- Accessing logcat programmatically from a regular app requires root access, which is not a viable solution for enterprise deployment

**Alternative:** Use the ApplicationExitInfo API (Android 11+) for crash/ANR monitoring of your own app, and Security/Network logs (Device Owner) for system-level events.

---

## Implementation Guidance

### Recommended Architecture for a Background Collector

**Core Components:**

1. **Foreground Service** (`TelemetryCollectorService`)
   - Runs continuously to ensure data collection is not terminated
   - Required for Android 8.0+ background restrictions
   - Must show persistent notification
   - Handles real-time event collection (battery, network changes)

2. **WorkManager for Periodic Tasks**
   - Use `PeriodicWorkRequest` for less time-sensitive data collection
   - Handles system-imposed scheduling constraints
   - Automatically retries on failures
   - Respects battery optimization and Doze mode

3. **Local Database** (Room recommended)
   - Store collected telemetry data locally
   - Enables offline operation
   - Supports batching and efficient uploads
   - Schema should match canonical telemetry format

4. **Data Upload Service**
   - Batched uploads to reduce network usage
   - Exponential backoff on failures
   - Respects network type (prefer WiFi)
   - Compress data before upload

**Architecture Diagram:**
```
┌─────────────────────────────────────────┐
│   TelemetryCollectorService (Foreground)│
│   - Battery BroadcastReceiver           │
│   - NetworkCallback                     │
│   - Real-time event handlers            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   WorkManager Periodic Tasks             │
│   - Storage stats collection            │
│   - App usage stats                     │
│   - Network stats                       │
│   - Security/compliance checks          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Local Database (Room)                  │
│   - TelemetryEvent table                │
│   - Buffered for batch upload           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Upload Service                         │
│   - Batch uploads                       │
│   - Retry logic                         │
│   - Compression                         │
└─────────────────────────────────────────┘
```

---

### Scheduling Strategy

**WorkManager Configuration:**

```java
// Periodic work for routine data collection
PeriodicWorkRequest collectWork = new PeriodicWorkRequest.Builder(
    TelemetryCollectorWorker.class,
    15, TimeUnit.MINUTES  // Minimum interval
)
.setConstraints(
    new Constraints.Builder()
        .setRequiredNetworkType(NetworkType.CONNECTED)
        .setRequiresBatteryNotLow(false) // Collect even on low battery
        .build()
)
.setBackoffCriteria(
    BackoffPolicy.EXPONENTIAL,
    WorkRequest.MIN_BACKOFF_MILLIS,
    TimeUnit.MILLISECONDS
)
.build();

WorkManager.getInstance(context).enqueue(collectWork);
```

**Foreground Service for Real-time Events:**

```java
public class TelemetryCollectorService extends Service {
    private BatteryReceiver batteryReceiver;
    private ConnectivityManager.NetworkCallback networkCallback;
    
    @Override
    public void onCreate() {
        super.onCreate();
        startForeground(NOTIFICATION_ID, createNotification());
        registerBatteryReceiver();
        registerNetworkCallback();
    }
    
    private void registerBatteryReceiver() {
        IntentFilter filter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        batteryReceiver = new BatteryReceiver();
        registerReceiver(batteryReceiver, filter);
    }
}
```

**Recommended Collection Intervals:**

| Datapoint Category | Collection Frequency | Method |
|-------------------|---------------------|---------|
| Battery Status | Event-driven (on change) | BroadcastReceiver |
| Network Connectivity | Event-driven (on change) | NetworkCallback |
| Network Stats | Every 1 hour | WorkManager |
| App Usage Stats | Every 1 hour | WorkManager |
| Storage Stats | Every 6 hours | WorkManager |
| Security Compliance | Every 12 hours | WorkManager |
| Device Info | Once at startup | Initialization |
| Security/Network Logs | Every 1 hour (when available) | WorkManager |

---

### Battery Impact Controls and Sampling Strategy

**Adaptive Sampling:**

```java
public class AdaptiveSampler {
    private long baseIntervalMs;
    private float batteryLevel;
    private boolean isCharging;
    
    public long getAdjustedInterval() {
        long interval = baseIntervalMs;
        
        // Increase interval when battery is low
        if (batteryLevel < 20 && !isCharging) {
            interval *= 2; // Double interval
        } else if (batteryLevel < 10 && !isCharging) {
            interval *= 4; // Quadruple interval
        }
        
        // Decrease interval when charging (can collect more frequently)
        if (isCharging) {
            interval = Math.max(interval / 2, baseIntervalMs / 2);
        }
        
        return interval;
    }
}
```

**Doze Mode and App Standby:**

- WorkManager automatically handles Doze mode and app standby
- Foreground services continue to operate but should minimize activity
- Consider reducing collection frequency when device is in Doze

**Battery Optimization Exemption:**

Device Owner apps can request battery optimization exemption (but should use judiciously):

```java
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
    PowerManager pm = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
    String packageName = context.getPackageName();
    if (!pm.isIgnoringBatteryOptimizations(packageName)) {
        // Request exemption (requires user action or Device Owner)
        Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
        intent.setData(Uri.parse("package:" + packageName));
        context.startActivity(intent);
    }
}
```

**Best Practices:**
- Batch operations together
- Use efficient data structures
- Minimize wake locks (WorkManager handles this)
- Avoid unnecessary CPU-intensive operations
- Compress data before storage/upload

---

### Local Buffering and Upload Batching

**Database Schema (Room):**

```java
@Entity(tableName = "telemetry_events")
public class TelemetryEvent {
    @PrimaryKey(autoGenerate = true)
    public long id;
    
    public long timestamp;
    public String eventType; // "battery", "network", "storage", etc.
    public String deviceId;
    public String jsonData; // JSON blob of event data
    public boolean uploaded;
    public long uploadTimestamp;
}
```

**Buffering Strategy:**

```java
@Dao
public interface TelemetryEventDao {
    @Query("SELECT * FROM telemetry_events WHERE uploaded = 0 ORDER BY timestamp ASC LIMIT :limit")
    List<TelemetryEvent> getPendingEvents(int limit);
    
    @Update
    void markAsUploaded(TelemetryEvent... events);
    
    @Query("DELETE FROM telemetry_events WHERE uploaded = 1 AND timestamp < :beforeTimestamp")
    void deleteOldUploadedEvents(long beforeTimestamp);
}
```

**Upload Batching:**

```java
public class TelemetryUploader {
    private static final int BATCH_SIZE = 100;
    private static final long MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24 hours
    
    public void uploadPendingEvents() {
        List<TelemetryEvent> events = dao.getPendingEvents(BATCH_SIZE);
        
        if (events.isEmpty()) {
            return;
        }
        
        // Compress and upload
        String json = gson.toJson(events);
        byte[] compressed = compress(json);
        
        uploadToServer(compressed, new Callback() {
            @Override
            public void onSuccess() {
                // Mark as uploaded
                for (TelemetryEvent event : events) {
                    event.uploaded = true;
                    event.uploadTimestamp = System.currentTimeMillis();
                }
                dao.updateEvents(events.toArray(new TelemetryEvent[0]));
                
                // Clean up old uploaded events
                long cutoff = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000); // 7 days
                dao.deleteOldUploadedEvents(cutoff);
            }
            
            @Override
            public void onFailure() {
                // Retry later (WorkManager will handle)
            }
        });
    }
}
```

**Upload Triggers:**

1. **Periodic:** WorkManager job every 4-6 hours
2. **Network Condition:** When WiFi is connected
3. **Batch Size:** When buffer reaches threshold (e.g., 100 events)
4. **Age-based:** Events older than 24 hours are prioritized

---

### Data Normalization into Canonical Schema

**Canonical Telemetry Schema:**

```json
{
  "deviceId": "string",
  "timestamp": "ISO8601",
  "eventType": "battery|network|storage|app_usage|security|device_info",
  "eventCategory": "string",
  "metrics": {
    // Type-specific metrics
  },
  "metadata": {
    "osVersion": "string",
    "sdkVersion": "int",
    "appVersion": "string",
    "collectionMethod": "polling|event|callback"
  }
}
```

**Type-Specific Schemas:**

**Battery Event:**
```json
{
  "eventType": "battery",
  "eventCategory": "status_change|periodic",
  "metrics": {
    "level": 85,
    "scale": 100,
    "status": "charging|discharging|full|not_charging",
    "health": "good|overheat|dead",
    "plugged": "usb|ac|wireless|unplugged",
    "temperature": 320,
    "voltage": 4200
  }
}
```

**Network Event:**
```json
{
  "eventType": "network",
  "eventCategory": "connectivity_change|usage_stats|signal_strength",
  "metrics": {
    "connected": true,
    "type": "wifi|cellular|ethernet",
    "wifiRssi": -65,
    "cellularType": "LTE",
    "cellularSignalStrength": -85,
    "bytesRx": 1024000,
    "bytesTx": 512000,
    "duration": 3600000
  }
}
```

**Storage Event:**
```json
{
  "eventType": "storage",
  "eventCategory": "capacity_check",
  "metrics": {
    "totalBytes": 64000000000,
    "availableBytes": 32000000000,
    "usedBytes": 32000000000,
    "freeBytes": 32000000000
  }
}
```

**App Usage Event:**
```json
{
  "eventType": "app_usage",
  "eventCategory": "usage_stats",
  "metrics": {
    "packageName": "com.example.app",
    "totalTimeInForeground": 3600000,
    "lastTimeUsed": 1640000000000,
    "launchCount": 5
  }
}
```

**Security Event:**
```json
{
  "eventType": "security",
  "eventCategory": "compliance_check|security_log",
  "metrics": {
    "encryptionStatus": "active",
    "passwordQuality": "complex",
    "securityPatchLevel": "2024-01-01",
    "deviceSecure": true,
    "biometricAvailable": true
  }
}
```

**Normalization Utility:**

```java
public class TelemetryNormalizer {
    public TelemetryEvent normalizeBatteryData(Intent batteryIntent) {
        TelemetryEvent event = new TelemetryEvent();
        event.deviceId = getDeviceId();
        event.timestamp = System.currentTimeMillis();
        event.eventType = "battery";
        event.eventCategory = "status_change";
        
        BatteryMetrics metrics = new BatteryMetrics();
        metrics.level = batteryIntent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
        metrics.scale = batteryIntent.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
        // ... populate other metrics
        
        event.metrics = metrics;
        event.metadata = createMetadata();
        
        return event;
    }
    
    // Similar methods for other event types...
}
```

---

## Safe MVP Collector List

The following datapoints are recommended for initial implementation, as they are accessible on most enterprise-managed devices running Android 8.0+ with minimal permissions and high reliability:

### Core Datapoints (No Special Permissions)

1. **Device Information**
   - Device model and manufacturer
   - OS version and SDK level
   - Security patch level (Android 6.0+)
   - Build fingerprint

2. **Battery Status**
   - Battery level and percentage
   - Charging status (charging/discharging/full)
   - Power source (USB/AC/wireless)
   - Battery health status

3. **Storage Information**
   - Available and total storage (internal)
   - Storage usage trends

4. **Network Connectivity Status**
   - Connected/disconnected state
   - Active network type (WiFi/cellular/ethernet)

5. **Security Compliance (Device Owner)**
   - Encryption status
   - Screen lock status (API 23+ via KeyguardManager)
   - Password quality (Device Owner)

### Datapoints Requiring User Grant (Can be Auto-granted by Device Owner)

6. **Network Usage Statistics** (PACKAGE_USAGE_STATS)
   - Data usage per app
   - Total data usage

7. **App Usage Statistics** (PACKAGE_USAGE_STATS)
   - Foreground app time
   - App launch counts

8. **Installed Applications** (QUERY_ALL_PACKAGES on Android 11+, exempt for Device Owner)
   - App inventory
   - App versions

### MVP Implementation Priority

**Phase 1 (Week 1-2):**
- Device information
- Battery status (event-driven)
- Storage information (periodic)
- Network connectivity status (event-driven)

**Phase 2 (Week 3-4):**
- Security compliance checks
- Network usage statistics
- App usage statistics
- Installed applications inventory

**Phase 3 (Week 5+):**
- Data normalization and upload
- Error handling and retry logic
- Performance optimization

---

## Device Owner Enhanced List

The following datapoints are available exclusively or with enhanced access when operating as a Device Owner:

### Exclusive Device Owner Features

1. **Security Logs** (API 24+)
   - Security-relevant events
   - Policy changes
   - Screen unlock attempts
   - Certificate installations

2. **Network Logs** (API 26+)
   - DNS lookup events
   - Network connection events
   - Per-app network activity

3. **Enhanced Permissions**
   - Auto-grant PACKAGE_USAGE_STATS to self
   - Query all packages without QUERY_ALL_PACKAGES permission
   - Access device serial number without user approval

4. **Policy Compliance Monitoring**
   - Password quality requirements
   - Encryption status enforcement
   - App restrictions status
   - User restrictions

5. **Device Identifiers**
   - Serial number access (API 26+)
   - IMEI/MEID access (with READ_PHONE_STATE, more reliable for Device Owner)

### Implementation Notes for Device Owner Features

**Enable Security/Network Logging:**

```java
DevicePolicyManager dpm = (DevicePolicyManager) context.getSystemService(Context.DEVICE_POLICY_SERVICE);
ComponentName admin = new ComponentName(context, DeviceAdminReceiver.class);

// Enable security logging (API 24+)
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
    dpm.setSecurityLoggingEnabled(admin, true);
}

// Enable network logging (API 26+)
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
    dpm.setNetworkLoggingEnabled(admin, true);
}
```

**Auto-grant Usage Stats Permission:**

```java
// Grant PACKAGE_USAGE_STATS to self (Device Owner only)
dpm.setPermissionGrantState(
    admin,
    context.getPackageName(),
    Manifest.permission.PACKAGE_USAGE_STATS,
    DevicePolicyManager.PERMISSION_GRANT_STATE_GRANTED
);
```

**Query All Packages (Device Owner exempt from restrictions):**

```java
// Device Owner apps can query all packages without QUERY_ALL_PACKAGES permission
PackageManager pm = context.getPackageManager();
List<ApplicationInfo> allApps = pm.getInstalledApplications(0); // Works for Device Owner
```

---

## Red List - Infeasible/Restricted Datapoints

The following datapoints are **not feasible** or **not allowed** on modern Android (API 26+) for third-party apps, including Device Owner apps:

### 1. Full System Logcat Access

**Why Not Available:**
- `READ_LOGS` permission is signature-level (system apps only)
- Requires platform key signature
- Cannot be granted to Device Owner or any third-party app
- Root access is not a viable enterprise solution

**Alternative:** Use ApplicationExitInfo (API 30+) for own app crashes, Security/Network logs (Device Owner) for system events

---

### 2. Real MAC Address (Android 8.0+)

**Why Not Available:**
- Starting Android 8.0, apps can only access randomized MAC addresses
- Real hardware MAC requires system app privileges
- WiFi MAC address access may return "02:00:00:00:00:00" (randomized)

**Alternative:** Use Android ID or device serial number for device identification

---

### 3. Complete Process/Service Enumeration

**Why Limited:**
- `getRunningAppProcesses()` only returns own app's processes on Android 8+
- Cannot enumerate all system processes
- Requires system app privileges for comprehensive monitoring

**Alternative:** Use UsageStatsManager for app activity, ApplicationExitInfo for own app processes

---

### 4. SMS and Call Logs

**Why Restricted:**
- Requires `READ_SMS` or `READ_CALL_LOG` permissions
- User must explicitly grant these permissions
- Cannot be auto-granted by Device Owner
- Highly sensitive privacy data

**Enterprise Note:** Even Device Owner cannot access SMS/call logs without explicit user consent. This is by design for privacy protection.

---

### 5. Precise Location Data (Without Consent)

**Why Restricted:**
- Requires `ACCESS_FINE_LOCATION` or `ACCESS_COARSE_LOCATION`
- Runtime permission required (user must grant)
- Cannot be auto-granted by Device Owner
- Location services must be enabled

**Note:** Device Owner can check if location services are enabled, but cannot access actual location coordinates without user consent or appropriate use case justification.

---

### 6. Clipboard Data

**Why Not Available:**
- Clipboard access is restricted to the app currently in focus
- Cannot monitor clipboard from background
- System apps may have limited access

**Alternative:** Not applicable for telemetry use cases

---

### 7. Keystroke Logging / Input Monitoring

**Why Not Available:**
- Requires Accessibility Service (user must explicitly enable)
- Highly invasive and privacy-sensitive
- Not suitable for general telemetry

**Enterprise Note:** Accessibility services require explicit user enablement and are intended for accessibility use cases, not telemetry.

---

### 8. Screen Content / Screenshots

**Why Restricted:**
- Requires `READ_FRAME_BUFFER` (system apps only) or MediaProjection API (user consent required)
- Cannot be automated without user interaction
- Highly privacy-sensitive

**Alternative:** Not applicable for telemetry use cases

---

### 9. Other Apps' Crash/ANR Data

**Why Limited:**
- `ApplicationExitInfo` (API 30+) only returns data for own app
- System apps may have broader access
- Third-party crash reporting SDKs require integration by each app

**Alternative:** Security logs (Device Owner) may capture some system-level crash information

---

### 10. Non-Resettable Device Identifiers (Restricted Access)

**Why Restricted:**
- IMEI/MEID access requires `READ_PHONE_STATE` and Device Owner context
- Android ID resets on factory reset and varies per app signing key
- Serial number access restricted on Android 8+ (Device Owner can access)

**Note:** While Device Owner can access some identifiers, they should be used judiciously for privacy compliance.

---

## References and Citations

### Official Android Documentation

1. **Device Policy Manager**
   - [DevicePolicyManager API Reference](https://developer.android.com/reference/android/app/admin/DevicePolicyManager)
   - [Enterprise Device Management](https://developer.android.com/work/dpc)
   - [Security and Network Logging](https://developer.android.com/work/dpc/logging)

2. **Usage Statistics**
   - [UsageStatsManager API Reference](https://developer.android.com/reference/android/app/usage/UsageStatsManager)
   - [NetworkStatsManager API Reference](https://developer.android.com/reference/android/app/usage/NetworkStatsManager)

3. **Battery Management**
   - [BatteryManager API Reference](https://developer.android.com/reference/android/os/BatteryManager)
   - [Power Management Best Practices](https://developer.android.com/training/monitoring-device-state/index.html)

4. **Network and Connectivity**
   - [ConnectivityManager API Reference](https://developer.android.com/reference/android/net/ConnectivityManager)
   - [Network Capabilities and Transports](https://developer.android.com/reference/android/net/NetworkCapabilities)

5. **Background Work**
   - [WorkManager Guide](https://developer.android.com/topic/libraries/architecture/workmanager)
   - [Foreground Services](https://developer.android.com/guide/components/foreground-services)
   - [Background Execution Limits](https://developer.android.com/about/versions/oreo/background)

6. **Permissions**
   - [Permission Overview](https://developer.android.com/guide/topics/permissions/overview)
   - [Runtime Permissions](https://developer.android.com/training/permissions/requesting)
   - [Special Permissions](https://developer.android.com/reference/android/Manifest.permission#PACKAGE_USAGE_STATS)

7. **Application Exit Information**
   - [ApplicationExitInfo API Reference](https://developer.android.com/reference/android/app/ApplicationExitInfo)
   - [Process Exit Reasons](https://developer.android.com/reference/android/app/ApplicationExitInfo.Reason)

8. **Privacy and Security**
   - [Android Privacy Best Practices](https://developer.android.com/privacy/best-practices)
   - [Device Identifiers](https://developer.android.com/training/articles/user-data-ids)
   - [Privacy Changes by Version](https://developer.android.com/about/versions/privacy)

### Android Version-Specific Documentation

- [Android 8.0 (API 26) Changes](https://developer.android.com/about/versions/oreo/android-8.0)
- [Android 9.0 (API 28) Changes](https://developer.android.com/about/versions/pie/android-9.0)
- [Android 10 (API 29) Privacy Changes](https://developer.android.com/about/versions/10/privacy/changes)
- [Android 11 (API 30) Changes](https://developer.android.com/about/versions/11)

### Enterprise Management

- [Android Enterprise Overview](https://developer.android.com/work)
- [Device Owner Setup](https://developer.android.com/work/dpc/build-device-owner)
- [Profile Owner vs Device Owner](https://developer.android.com/work/dpc/device-owner)

---

## Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial catalog for Android 8.0+ Device Owner context |

---

## Appendix: Quick Reference Matrix

| Datapoint | API Class | Min API | Permission | Device Owner Required | Collection Method |
|-----------|-----------|---------|------------|----------------------|-------------------|
| Device Model | Build | 1 | None | No | Polling |
| OS Version | Build | 1 | None | No | Polling |
| Security Patch | Build | 23 | None | No | Polling |
| Serial Number | Build | 26 | READ_PHONE_STATE | Yes (recommended) | Polling |
| Battery Level | BatteryManager | 1 | None | No | Event/Polling |
| Battery Health | BatteryManager | 1 | None | No | Event/Polling |
| Network Status | ConnectivityManager | 1 | ACCESS_NETWORK_STATE | No | Event/Polling |
| WiFi RSSI | WifiManager | 1 | ACCESS_WIFI_STATE | No | Polling |
| Cellular Signal | TelephonyManager | 1 | READ_PHONE_STATE | Recommended | Polling |
| Network Stats | NetworkStatsManager | 23 | PACKAGE_USAGE_STATS | Can auto-grant | Polling |
| Storage Info | StatFs | 1 | None (internal) | No | Polling |
| Storage by App | StorageStatsManager | 26 | PACKAGE_USAGE_STATS | Can auto-grant | Polling |
| Memory Info | ActivityManager | 1 | None | No | Polling |
| Installed Apps | PackageManager | 1 | QUERY_ALL_PACKAGES (11+) | Exempt if DO | Polling |
| App Usage | UsageStatsManager | 21 | PACKAGE_USAGE_STATS | Can auto-grant | Polling |
| Encryption Status | DevicePolicyManager | 11 | BIND_DEVICE_ADMIN | Yes | Polling |
| Password Quality | DevicePolicyManager | 8 | BIND_DEVICE_ADMIN | Yes | Polling |
| Security Logs | DevicePolicyManager | 24 | Device Owner | Yes | Event-driven |
| Network Logs | DevicePolicyManager | 26 | Device Owner | Yes | Event-driven |
| App Crashes | ApplicationExitInfo | 30 | None (own app) | No | Polling |

**Legend:**
- **Min API:** Minimum Android API level required
- **Device Owner Required:** Whether Device Owner role is required or just recommended
- **Collection Method:** Event-driven (real-time) vs Polling (periodic)

---

*End of Android Datapoints Catalog*

