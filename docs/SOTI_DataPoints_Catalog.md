# SOTI XSight & MobiControl Data Points Research Catalog

**Research Date:** December 2024  
**Status:** Based on public documentation - requires validation with SOTI SMEs  
**Target Products:** SOTI XSight, SOTI MobiControl (versions TBD)

---

## Executive Summary

This catalog compiles telemetry and management datapoints available from SOTI XSight and SOTI MobiControl for anomaly detection use cases. Data points are categorized by domain, marked with availability confidence levels, and include integration guidance.

**Key Findings:**
- **MobiControl** provides comprehensive device inventory, compliance, and management data via UI and REST API
- **XSight** focuses on real-time analytics, performance metrics, and device health monitoring
- Many datapoints require joining across both systems for complete device context
- Some advanced metrics may require custom device agent collection

---

## 1. Device Identity and Hardware

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Device ID | Unique identifier assigned by MobiControl | MobiControl | **Confirmed** | UI, API | Periodic | Device tracking, enrollment management |
| Device Name | The assigned name of the device | MobiControl | **Confirmed** | UI, API | Periodic | Unauthorized device name changes |
| Device Model | Model name/number of the device | MobiControl | **Confirmed** | UI, API | Periodic | Unsupported device models in fleet |
| Serial Number | Manufacturer-assigned serial number | MobiControl | **Confirmed** | UI, API | Periodic | Device authentication, warranty tracking |
| IMEI | International Mobile Equipment Identity number | MobiControl | **Confirmed** | UI, API | Periodic | Stolen device detection, cellular identification |
| MEID | Mobile Equipment Identifier (CDMA devices) | MobiControl | **Likely** | UI, API | Periodic | CDMA device identification |
| MAC Address | Media Access Control address of network interface | MobiControl | **Confirmed** | UI, API | Periodic | Network security audits, duplicate detection |
| Manufacturer | Device manufacturer name | MobiControl | **Confirmed** | UI, API | Periodic | Unauthorized device detection |
| Device Family | Category (smartphone, tablet, rugged) | MobiControl | **Confirmed** | UI, API | Periodic | Policy application by device type |
| Device Kind | Type classification | MobiControl | **Confirmed** | UI, API | Periodic | Device categorization |
| Processor Type | Type of CPU in the device | MobiControl | **Confirmed** | UI, API | Periodic | Performance assessment, compatibility |
| RAM Size | Amount of Random Access Memory installed | MobiControl | **Confirmed** | UI, API | Periodic | Insufficient memory detection |
| Storage Capacity | Total internal storage capacity | MobiControl | **Confirmed** | UI, API | Periodic | Storage planning, capacity issues |
| Hardware Version | Version of device hardware | MobiControl | **Confirmed** | UI, API | Periodic | Hardware lifecycle management |

**Notes:**
- Device ID, Serial Number, and IMEI are primary keys for joining data across systems
- Multi-tenancy: Device IDs may be scoped by tenant/site in enterprise deployments

---

## 2. OS and Patch Level

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| OS Version | Operating system version installed | MobiControl | **Confirmed** | UI, API | Periodic | Outdated OS detection, compatibility |
| OS Build Number | Specific build number of the OS | MobiControl | **Confirmed** | UI, API | Periodic | Specific OS version identification |
| Security Patch Level | Date/level of last security patch applied | MobiControl | **Confirmed** | UI, API | Periodic | Vulnerability assessment, compliance |
| Kernel Version | Version of the OS kernel | MobiControl | **Confirmed** | UI, API | Periodic | Outdated kernel detection |
| Android Security Patch Date | Android-specific patch date | MobiControl | **Confirmed** | UI, API | Periodic | Android vulnerability compliance |
| iOS Version | iOS version (for Apple devices) | MobiControl | **Confirmed** | UI, API | Periodic | iOS compatibility checks |

**Notes:**
- Patch level data is critical for security anomaly detection
- Update cadence depends on device check-in frequency

---

## 3. Battery and Power

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Battery Level | Current battery charge percentage | XSight, MobiControl | **Confirmed** | UI, API (XSight), UI (MobiControl) | Near real-time (XSight), Periodic (MobiControl) | Battery degradation, rapid depletion |
| Battery Health | Overall health status of the battery | XSight | **Confirmed** | UI | Near real-time | Predictive battery failure |
| Charging Status | Indicates if device is currently charging | XSight, MobiControl | **Confirmed** | UI | Near real-time (XSight), Periodic (MobiControl) | Charging issues, usage patterns |
| Battery Temperature | Battery temperature reading | XSight | **Likely** | UI | Near real-time | Overheating detection |
| Power Source | AC, USB, wireless charging | MobiControl | **Likely** | UI, API | Periodic | Charging method compliance |

**Notes:**
- XSight provides more real-time battery metrics than MobiControl
- Battery health degradation patterns are valuable for predictive maintenance

---

## 4. Network and Connectivity

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Wi-Fi SSID | Connected Wi-Fi network name | MobiControl | **Confirmed** | UI, API | Periodic | Unauthorized network connections |
| Wi-Fi Signal Strength | Strength of Wi-Fi signal (dBm) | MobiControl | **Confirmed** | UI, API | Periodic | Connectivity issues, coverage gaps |
| Wi-Fi MAC Address | MAC address of Wi-Fi interface | MobiControl | **Confirmed** | UI, API | Periodic | Network interface identification |
| Cellular Carrier | Mobile network operator name | MobiControl | **Confirmed** | UI, API | Periodic | Roaming detection, carrier-specific issues |
| Cellular Signal Strength | Strength of cellular signal (dBm) | XSight | **Confirmed** | UI | Near real-time | Coverage issues, connectivity problems |
| Cellular Network Type | 4G, 5G, LTE, etc. | MobiControl | **Confirmed** | UI, API | Periodic | Network capability assessment |
| IP Address | Current IP address assigned to device | MobiControl | **Confirmed** | UI, API | Near real-time | Network security, unexpected IP detection |
| VPN Status | Indicates if VPN connection is active | MobiControl | **Confirmed** | UI, API | Near real-time | Secure connection compliance |
| VPN Type | Type of VPN connection (if applicable) | MobiControl | **Likely** | UI, API | Periodic | VPN configuration compliance |
| Data Usage | Cellular data consumption | MobiControl | **Likely** | UI, API | Periodic | Data overage detection, usage anomalies |
| Network Type | Wi-Fi, Cellular, Ethernet | MobiControl | **Confirmed** | UI, API | Periodic | Connectivity method tracking |

**Notes:**
- Network data is critical for connectivity anomaly detection
- VPN status is important for security compliance monitoring

---

## 5. Performance and Health

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| CPU Usage | Percentage of CPU resources in use | XSight | **Confirmed** | UI | Near real-time | Performance bottlenecks, high load detection |
| Memory Usage | Amount of RAM currently in use | XSight | **Confirmed** | UI | Near real-time | Memory leaks, insufficient RAM |
| Storage Usage | Amount of storage space used/available | XSight, MobiControl | **Confirmed** | UI | Near real-time (XSight), Periodic (MobiControl) | Storage capacity issues, disk full |
| Device Temperature | Current operating temperature of device | XSight | **Confirmed** | UI | Near real-time | Overheating issues, thermal anomalies |
| CPU Temperature | CPU-specific temperature | XSight | **Likely** | UI | Near real-time | CPU thermal throttling detection |
| Available Memory | Free RAM available | XSight | **Confirmed** | UI | Near real-time | Memory pressure detection |
| Storage Available | Free storage space | XSight, MobiControl | **Confirmed** | UI | Near real-time (XSight), Periodic (MobiControl) | Low storage warnings |
| CPU Cores | Number of CPU cores | MobiControl | **Confirmed** | UI, API | Periodic | Hardware capability assessment |
| CPU Frequency | Current CPU frequency | XSight | **Likely** | UI | Near real-time | Performance throttling detection |

**Notes:**
- XSight excels at real-time performance metrics
- Temperature data is valuable for hardware failure prediction

---

## 6. App Inventory and Versions

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Installed Applications | List of applications installed on device | MobiControl | **Confirmed** | UI, API | Periodic | Unauthorized app detection |
| Application Versions | Version numbers of installed applications | MobiControl | **Confirmed** | UI, API | Periodic | Outdated app detection, vulnerability assessment |
| Running Applications | Currently running applications | XSight | **Confirmed** | UI | Near real-time | Unauthorized app execution |
| App Usage | Metrics on application usage patterns | XSight | **Confirmed** | UI, API | Periodic | Underutilized/overutilized apps |
| App Install Date | Date application was installed | MobiControl | **Likely** | UI, API | Periodic | App lifecycle tracking |
| App Size | Size of installed applications | MobiControl | **Likely** | UI, API | Periodic | Storage impact analysis |
| System Apps | System vs user-installed apps | MobiControl | **Confirmed** | UI, API | Periodic | System app modification detection |

**Notes:**
- App inventory is critical for compliance and security anomaly detection
- Version mismatches can indicate update failures or policy violations

---

## 7. Security Posture

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Encryption Status | Indicates if device storage is encrypted | MobiControl | **Confirmed** | UI, API | Periodic | Data protection compliance |
| Passcode Status | Indicates if passcode is set on device | MobiControl | **Confirmed** | UI, API | Periodic | Unauthorized access prevention |
| Passcode Complexity | Complexity requirements met | MobiControl | **Likely** | UI, API | Periodic | Weak passcode detection |
| Jailbreak/Root Status | Indicates if device is jailbroken or rooted | MobiControl | **Confirmed** | UI, API | Periodic | Security compromise detection |
| Developer Mode | Developer options enabled | MobiControl | **Confirmed** | UI, API | Periodic | Unauthorized developer access |
| USB Debugging | USB debugging enabled | MobiControl | **Confirmed** | UI, API | Periodic | Security policy violation |
| Unknown Sources | Installation from unknown sources allowed | MobiControl | **Confirmed** | UI, API | Periodic | Security risk detection |
| Compliance Status | Overall compliance status based on policies | MobiControl | **Confirmed** | UI, API | Periodic | Policy enforcement, compliance violations |
| Applied Policies | List of security policies applied to device | MobiControl | **Confirmed** | UI, API | Periodic | Policy enforcement verification |
| Screen Lock Type | Type of screen lock (PIN, pattern, biometric) | MobiControl | **Confirmed** | UI, API | Periodic | Security policy compliance |
| Biometric Enrollment | Biometric authentication enrolled | MobiControl | **Likely** | UI, API | Periodic | Multi-factor authentication status |
| Certificate Status | Status of installed certificates | MobiControl | **Likely** | UI, API | Periodic | Certificate expiration, trust issues |

**Notes:**
- Security posture data is essential for compliance anomaly detection
- Jailbreak/root status is a critical security indicator

---

## 8. Location and Geofencing

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| GPS Location | Current GPS coordinates (latitude, longitude) | XSight, MobiControl | **Confirmed** | UI | Near real-time | Asset tracking, unauthorized movement |
| Location Accuracy | Accuracy of GPS reading (meters) | XSight | **Likely** | UI | Near real-time | Location data quality assessment |
| Location Timestamp | Time of last location update | XSight, MobiControl | **Confirmed** | UI | Near real-time | Stale location detection |
| Geofence Status | Indicates if device is within defined geofence | XSight, MobiControl | **Confirmed** | UI | Near real-time | Unauthorized location detection |
| Geofence Events | Entry/exit events from geofences | MobiControl | **Confirmed** | UI, API | Event-driven | Location policy violations |
| Location History | Historical location data | MobiControl | **Confirmed** | UI, API | Periodic | Movement pattern analysis |
| Last Known Location | Most recent location before going offline | MobiControl | **Confirmed** | UI, API | Periodic | Offline device tracking |

**Notes:**
- Location data may be sensitive - consider privacy implications
- Geofencing is valuable for asset protection and compliance

---

## 9. Remote Actions and Management Events

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Remote Control Sessions | Logs of remote control sessions initiated | MobiControl, XSight | **Confirmed** | UI, API | Event-driven | Support activity auditing, unauthorized access |
| Device Commands | History of commands sent to device | MobiControl | **Confirmed** | UI, API | Event-driven | Configuration change tracking |
| Device Reboots | Records of device reboot events | MobiControl | **Confirmed** | UI, API | Event-driven | Unexpected reboot detection |
| Device Wipe Events | Records of remote wipe commands executed | MobiControl | **Confirmed** | UI, API | Event-driven | Security incident tracking |
| Configuration Changes | Records of configuration changes applied | MobiControl | **Confirmed** | UI, API | Event-driven | Unauthorized configuration changes |
| App Installation Events | Records of app installations | MobiControl | **Confirmed** | UI, API | Event-driven | Unauthorized app installation |
| App Uninstallation Events | Records of app uninstallations | MobiControl | **Confirmed** | UI, API | Event-driven | Critical app removal detection |
| Policy Application Events | Records of policy applications | MobiControl | **Confirmed** | UI, API | Event-driven | Policy enforcement tracking |
| Enrollment Events | Device enrollment history | MobiControl | **Confirmed** | UI, API | Event-driven | Enrollment anomalies |

**Notes:**
- Event logs are critical for audit trails and anomaly detection
- Remote control sessions should be monitored for security

---

## 10. Alerts and Incidents

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Device Alerts | Notifications of device issues or events | MobiControl, XSight | **Confirmed** | UI, API | Event-driven | Proactive issue resolution |
| Security Alerts | Notifications of security-related events | MobiControl | **Confirmed** | UI, API | Event-driven | Security incident response |
| Performance Alerts | Notifications of performance issues | XSight | **Confirmed** | UI, API | Event-driven | Performance degradation detection |
| Compliance Violations | Records of policy compliance violations | MobiControl | **Confirmed** | UI, API | Event-driven | Policy enforcement monitoring |
| Incident Reports | Detailed reports of incidents involving device | XSight, MobiControl | **Confirmed** | UI, API | Event-driven | Incident pattern analysis |
| Alert Severity | Severity level of alerts (critical, warning, info) | MobiControl, XSight | **Confirmed** | UI, API | Event-driven | Alert prioritization |
| Alert Timestamp | Time when alert was generated | MobiControl, XSight | **Confirmed** | UI, API | Event-driven | Alert correlation and timing analysis |

**Notes:**
- Alert data is valuable for anomaly detection model training
- Alert patterns can indicate systemic issues

---

## 11. Connectivity and Uptime

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Uptime | Duration since last device reboot | XSight, MobiControl | **Confirmed** | UI | Near real-time (XSight), Periodic (MobiControl) | System stability, unexpected reboots |
| Downtime | Duration device has been non-operational | XSight | **Confirmed** | UI | Near real-time | Service level monitoring |
| Last Connected Time | Timestamp of last connection to MobiControl | MobiControl | **Confirmed** | UI, API | Periodic | Offline device detection |
| Connection Duration | Duration of current connection session | MobiControl | **Likely** | UI, API | Periodic | Connection stability |
| Network Downtime | Periods when network was unavailable | XSight | **Confirmed** | UI | Near real-time | Connectivity issue analysis |
| Check-in Frequency | Frequency of device check-ins | MobiControl | **Confirmed** | UI, API | Calculated | Device communication anomalies |
| Last Check-in | Timestamp of last device check-in | MobiControl | **Confirmed** | UI, API | Periodic | Stale device detection |

**Notes:**
- Uptime data helps identify devices with stability issues
- Connection patterns can indicate network or device problems

---

## 12. Peripheral Information

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Connected Peripherals | List of peripherals connected to device | MobiControl | **Confirmed** | UI, API | Periodic | Peripheral compatibility, missing peripherals |
| Peripheral Status | Operational status of connected peripherals | MobiControl | **Confirmed** | UI, API | Periodic | Peripheral failure detection |
| Scanner Status | Status of barcode scanner peripherals | MobiControl | **Confirmed** | UI, API | Periodic | Scanner functionality issues |
| Bluetooth Status | Status of Bluetooth connectivity | MobiControl | **Confirmed** | UI, API | Periodic | Bluetooth accessory troubleshooting |
| Bluetooth Devices | List of paired Bluetooth devices | MobiControl | **Likely** | UI, API | Periodic | Unauthorized Bluetooth pairing |
| USB Devices | List of connected USB devices | MobiControl | **Likely** | UI, API | Periodic | Unauthorized USB device detection |
| NFC Status | Near Field Communication status | MobiControl | **Likely** | UI, API | Periodic | NFC functionality verification |

**Notes:**
- Peripheral data is important for rugged device deployments
- Scanner status is critical for warehouse/retail use cases

---

## 13. Custom Data and Extensibility

| Datapoint Name | Description | Source Product | Availability | Retrieval Path | Update Cadence | Typical Anomaly Use Cases |
|----------------|-------------|----------------|-------------|----------------|----------------|---------------------------|
| Custom Attributes | User-defined attributes for devices | MobiControl | **Confirmed** | UI, API | As configured | Custom reporting, business-specific metadata |
| Scripting Capabilities | Ability to execute custom scripts on devices | MobiControl | **Confirmed** | UI, API | As configured | Automated device management tasks |
| JavaScript Device Agent | Custom JavaScript execution on device | MobiControl | **Confirmed** | Device Agent | As configured | Custom data collection, automation |
| System Properties | Device system properties accessible via scripts | MobiControl | **Confirmed** | Device Agent Scripts | As configured | Extended device information collection |
| Custom Metrics | User-defined metrics collection | MobiControl | **Likely** | Device Agent | As configured | Business-specific anomaly detection |
| External API Integration | Integration with external systems for data | MobiControl | **Likely** | API | As configured | Cross-system data correlation |

**Notes:**
- Custom attributes and scripting enable extensibility
- JavaScript device agent can collect additional datapoints not natively available
- System properties may include device-specific information not in standard inventory

---

## Integration Implications

### Unified Telemetry Model

**Easiest to Unify:**
- Device Identity (Device ID, Serial Number, IMEI)
- OS Version and Patch Level
- Battery Level (both systems)
- Installed Applications
- Compliance Status

**Requires Joining:**
- Performance metrics (XSight) + Device inventory (MobiControl)
- Real-time analytics (XSight) + Policy compliance (MobiControl)
- Location data (both systems) + Geofencing policies (MobiControl)

### Key Identifiers for Joining

| Identifier | Source | Use Case |
|------------|--------|----------|
| Device ID | MobiControl | Primary key for device management |
| Serial Number | MobiControl | Hardware-level identification |
| IMEI | MobiControl | Cellular device identification |
| Enrollment ID | MobiControl | Enrollment-specific identifier |
| Tenant ID | Both | Multi-tenant data isolation |
| Site ID | Both | Site/location-based grouping |
| Device Name | Both | Human-readable identifier (may not be unique) |

**Joining Strategy:**
1. Primary: Device ID (MobiControl) → Device ID (XSight) if available
2. Fallback: Serial Number (both systems)
3. Alternative: IMEI (for cellular devices)

### Data Collection Gaps

**Gaps Requiring Custom Agent:**
- Real-time application-level metrics (beyond what XSight provides)
- Custom business metrics specific to use case
- Integration with third-party sensors or peripherals
- Device-specific telemetry not exposed via standard APIs

**Custom Agent Considerations:**
- Android Device Agent (MobiControl) supports JavaScript execution
- Can access system properties and device sensors
- May require additional permissions or device configuration
- Consider battery impact of custom data collection

### Multi-Tenancy Considerations

- Device IDs may be scoped by tenant in enterprise deployments
- API access may require tenant context
- Data isolation must be maintained in anomaly detection system
- Consider tenant-specific baselines for anomaly detection

---

## Recommended Datapoint Sets

### MVP Datapoint Set for Anomaly Detection

**Core Identity & Status:**
- Device ID, Serial Number, IMEI
- Device Model, Manufacturer
- OS Version, Security Patch Level
- Compliance Status

**Performance & Health:**
- Battery Level, Battery Health
- CPU Usage, Memory Usage, Storage Usage
- Device Temperature

**Connectivity:**
- Wi-Fi SSID, Cellular Signal Strength
- VPN Status
- Last Connected Time, Uptime

**Security:**
- Encryption Status, Passcode Status
- Jailbreak/Root Status

**Location:**
- GPS Location, Geofence Status

**Applications:**
- Installed Applications, Application Versions

**Rationale:** These datapoints provide comprehensive coverage for common anomaly patterns (performance degradation, security violations, connectivity issues, unauthorized changes) while being readily available from both systems.

### Phase 2 Datapoint Set for Advanced Detectors

**Extended Performance:**
- CPU Temperature, CPU Frequency
- Available Memory, Storage Available
- Network Downtime, Connection Duration

**Advanced Security:**
- Passcode Complexity, Screen Lock Type
- Biometric Enrollment, Certificate Status
- Developer Mode, USB Debugging

**Event Logs:**
- Remote Control Sessions
- Device Reboots, Configuration Changes
- App Installation/Uninstallation Events

**Peripherals:**
- Connected Peripherals, Peripheral Status
- Scanner Status, Bluetooth Devices

**Custom Data:**
- Custom Attributes
- Script-collected metrics via Device Agent

**Rationale:** Phase 2 adds depth for advanced anomaly detection including behavioral patterns, event correlation, and custom business metrics.

---

## Validation Steps and Questions for SOTI SMEs

### API and Data Access

1. **REST API Documentation**
   - Request detailed REST API documentation for both XSight and MobiControl
   - Confirm authentication methods (OAuth, API keys, etc.)
   - Verify rate limits and pagination strategies
   - Question: What API version should we target?

2. **Data Retrieval Methods**
   - Confirm which datapoints are available via REST API vs UI only
   - Verify database access options (if any) for bulk data extraction
   - Question: Can we access XSight data warehouse directly, or only via API/UI?

3. **Update Cadences**
   - Confirm actual update frequencies for each datapoint category
   - Understand real-time vs batch update mechanisms
   - Question: What is the typical latency for datapoint updates?

### Datapoint Availability

4. **Uncertain Datapoints**
   - Validate datapoints marked as "Likely" or "Uncertain"
   - Confirm availability of advanced metrics (CPU temperature, custom properties)
   - Question: Which datapoints require custom device agent scripts to collect?

5. **Version-Specific Features**
   - Understand datapoint availability by product version
   - Identify version-specific API endpoints or capabilities
   - Question: What versions of XSight and MobiControl are we targeting?

### Multi-Tenancy and Security

6. **Multi-Tenant Data Access**
   - Understand how tenant isolation works in API access
   - Confirm tenant ID and site ID availability in API responses
   - Question: How do we scope API queries by tenant/site?

7. **Data Sensitivity**
   - Identify which datapoints are considered sensitive (location, personal data)
   - Understand data retention policies
   - Question: Are there datapoints we should exclude or anonymize?

### Integration and Joining

8. **Device Identifier Mapping**
   - Confirm how to join XSight and MobiControl data
   - Verify Device ID consistency across systems
   - Question: Is there a canonical device identifier that works across both systems?

9. **Historical Data Access**
   - Understand how to access historical datapoint values
   - Confirm data retention periods
   - Question: How far back can we query historical device data?

### Custom Collection

10. **Device Agent Capabilities**
    - Understand JavaScript device agent capabilities and limitations
    - Confirm system properties accessible via scripts
    - Question: What additional datapoints can we collect via custom device agent scripts?

11. **Custom Attributes**
    - Understand custom attribute limits and types
    - Confirm API access to custom attributes
    - Question: Can we programmatically set/update custom attributes via API?

### Performance and Scale

12. **API Performance**
    - Understand recommended polling frequencies
    - Confirm bulk data export capabilities
    - Question: What is the recommended approach for ingesting data from thousands of devices?

13. **Data Volume**
    - Estimate data volumes for target device counts
    - Understand any data aggregation or sampling options
    - Question: Are there pre-aggregated views or summaries available?

---

## Source Links and Citations

### Official SOTI Documentation

1. **SOTI MobiControl REST API Documentation**
   - URL: https://soti.net/mc/help/v2025.1/en/adminutility/tools/restapi.html
   - Notes: REST API reference for MobiControl device management

2. **SOTI MobiControl Device Information**
   - URL: https://it.soti.net/mc/help/v14.1/en/console/start/device_information.html
   - Notes: Device details and inventory information

3. **SOTI MobiControl Device Agents**
   - URL: https://it.soti.net/mc/help/v15.0/en/console/devices/managing/enrollment/deviceagent.html
   - Notes: Device agent capabilities and JavaScript scripting

4. **SOTI MobiControl Device Details Tab**
   - URL: https://www.soti.net/mc/help/v2025.1/en/console/start/the_device_details_tab.html
   - Notes: UI-based device information reference

5. **SOTI XSight Product Information**
   - URL: https://www.soti.net/xsight
   - Notes: XSight product overview and capabilities

### Additional Resources

6. **SOTI MobiControl Device Logs**
   - URL: https://it.soti.net/mc/help/v15.0/en/console/troubleshooting/devicelogs/devicelogs.html
   - Notes: Device log access and troubleshooting

**Note:** All URLs were accessed in December 2024. Documentation may be updated - verify with latest SOTI documentation.

---

## Appendix: Data Point Availability Matrix

### Quick Reference by Source

| Category | MobiControl | XSight | Notes |
|----------|-------------|--------|-------|
| Device Identity | ✅ Comprehensive | ⚠️ Limited | MobiControl is primary source |
| OS & Patches | ✅ Comprehensive | ❌ Not available | MobiControl only |
| Battery | ✅ Basic | ✅ Advanced | XSight has real-time health metrics |
| Network | ✅ Comprehensive | ✅ Signal strength | Both systems complement each other |
| Performance | ❌ Limited | ✅ Real-time | XSight excels at performance metrics |
| Apps | ✅ Comprehensive | ✅ Running apps | MobiControl for inventory, XSight for runtime |
| Security | ✅ Comprehensive | ❌ Limited | MobiControl is primary source |
| Location | ✅ Comprehensive | ✅ Real-time | Both systems provide location data |
| Events | ✅ Comprehensive | ✅ Alerts | MobiControl for management events, XSight for alerts |
| Uptime | ✅ Basic | ✅ Advanced | XSight provides more detailed uptime metrics |
| Peripherals | ✅ Comprehensive | ❌ Not available | MobiControl only |
| Custom Data | ✅ Extensible | ❌ Not available | MobiControl via device agent |

**Legend:**
- ✅ Comprehensive/Advanced
- ⚠️ Limited/Basic
- ❌ Not available

---

## Next Steps

1. **Validate with SOTI SMEs** using the questions in the Validation Steps section
2. **Confirm target versions** of XSight and MobiControl
3. **Obtain API credentials** and test API access
4. **Build proof-of-concept** data ingestion for MVP datapoint set
5. **Design canonical telemetry model** based on validated datapoints
6. **Plan custom device agent** development if needed for Phase 2 datapoints

---

**Document Status:** Draft - Requires validation with SOTI internal documentation and SMEs  
**Last Updated:** December 2024  
**Maintained By:** Anomaly Detection Platform Team

