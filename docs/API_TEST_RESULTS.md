# SOTI MobiControl API Test Results

**Date:** December 2024  
**Server:** https://a0024952.mobicontrol.cloud  
**Status:** ✅ **SUCCESSFUL**

---

## Authentication

### ✅ OAuth 2.0 Password Grant Flow

**Endpoint:** `/MobiControl/api/token`

**Method:** POST  
**Content-Type:** `application/x-www-form-urlencoded`

**Request Body:**
```
grant_type=password
username=Yannick-API
password=Feyenoord010
client_id=65c1d00f44c847618149f0194501938c
client_secret=74GJyL0OVmnCEeGTBOpSFXlwx4siFHPcSX3uVbIfv44=
```

**Response:**
```json
{
  "access_token": "...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Status:** ✅ Working

---

## API Endpoints

### ✅ Devices Endpoint

**Endpoint:** `/MobiControl/api/devices`  
**Method:** GET  
**Authentication:** Bearer token

**Response Format:** Array of device objects (not paginated dict)

**Sample Response:**
```json
[
  {
    "$type": "DeviceAndroidForWork",
    "DeviceId": "...",
    "DeviceName": "...",
    "Manufacturer": "...",
    "Model": "...",
    "OSVersion": "...",
    "BatteryStatus": {...},
    "ComplianceStatus": "...",
    ...
  }
]
```

**Status:** ✅ Working  
**Devices Found:** 27 devices

---

## Available Device Data Fields

Based on the API response, the following fields are available:

### Identity & Hardware
- `DeviceId` - Unique device identifier
- `DeviceName` - Device name
- `Manufacturer` - Device manufacturer
- `Model` - Device model
- `HardwareSerialNumber` - Hardware serial number
- `MobileSerialNumber` - Mobile serial number
- `IMEI_MEID_ESN` - IMEI/MEID/ESN
- `ICCID` - SIM card identifier
- `MACAddress` - MAC address
- `BluetoothMAC` - Bluetooth MAC address
- `WifiMAC` - WiFi MAC address
- `HardwareVersion` - Hardware version
- `OEMVersion` - OEM version

### Operating System
- `OSVersion` - Operating system version
- `BuildVersion` - Build version
- `AndroidApiLevel` - Android API level
- `BuildSecurityPatch` - Security patch level
- `Platform` - Platform type

### Battery & Power
- `BatteryStatus` - Battery status object
- `IsCharging` - Charging status

### Network & Connectivity
- `NetworkSSID` - WiFi SSID
- `NetworkRSSI` - WiFi signal strength
- `NetworkBSSID` - WiFi BSSID
- `NetworkConnectionType` - Connection type
- `CellularCarrier` - Cellular carrier
- `CellularSignalStrength` - Cellular signal strength
- `CellularTechnology` - Cellular technology (LTE, 5G, etc.)
- `SIMCarrierNetwork` - SIM carrier network
- `InRoaming` - Roaming status
- `VpnIp` - VPN IP address
- `VpnIPv6` - VPN IPv6 address
- `Ipv6` - IPv6 support

### Performance & Health
- `Memory` - Memory information object
- `AgentVersion` - Agent version
- `PlugInVersion` - Plugin version

### Security & Compliance
- `ComplianceStatus` - Overall compliance status
- `CompliancePolicyStatus` - Policy compliance status
- `ComplianceItems` - List of compliance items
- `IsEncrypted` - Encryption status
- `HardwareEncryption` - Hardware encryption status
- `HardwareEncryptionCaps` - Hardware encryption capabilities
- `PasscodeEnabled` - Passcode enabled
- `PasscodeStatus` - Passcode status
- `IsOSSecure` - OS security status
- `SafetynetAttestationStatus` - SafetyNet attestation
- `AmapiAttestationStatus` - AMAPI attestation

### Applications & Enterprise
- `AndroidForWork` - Android for Work configuration
- `AndroidEnterpriseEmail` - Enterprise email
- `AndroidAccountType` - Account type
- `AndroidDeviceAdmin` - Device admin status
- `UserIdentities` - User identities
- `UserAccountsCount` - User account count

### Location & Time
- `TimeZone` - Device timezone
- `LastCheckInTime` - Last check-in timestamp
- `LastAgentConnectTime` - Last agent connection
- `LastAgentDisconnectTime` - Last agent disconnection

### Management
- `EnrollmentType` - Enrollment type
- `EnrollmentTime` - Enrollment timestamp
- `IsAgentOnline` - Agent online status
- `IsAgentCompatible` - Agent compatibility
- `AgentCompatibilityStatus` - Compatibility status
- `RemoteControlSessionStatus` - Remote control status
- `DateAndTimeOfRemoteControlSessionStart` - Remote control session start

### Custom Data
- `CustomData` - Custom data object
- `CustomAttributes` - Custom attributes

### Additional Fields
- `Kind` - Device kind
- `Family` - Device family
- `Path` - Device path
- `ServerName` - Server name
- `HostName` - Host name
- `PhoneNumber` - Phone number
- `PersonalizedName` - Personalized name
- `Language` - Device language
- `AzureRegistrationMode` - Azure registration mode
- `AzureRegistrationStatus` - Azure registration status
- `AzureDeviceId` - Azure device ID
- `AzureComplianceStatus` - Azure compliance status
- `AzureUserId` - Azure user ID

---

## Implementation Status

### ✅ Completed
- [x] OAuth 2.0 authentication (password grant)
- [x] Basic authentication fallback
- [x] Device list retrieval
- [x] API client library
- [x] Test script

### 🔄 In Progress
- [ ] Single device details endpoint
- [ ] Device applications endpoint
- [ ] Device events endpoint
- [ ] Pagination support (if available)
- [ ] Filtering and query parameters

### 📋 To Do
- [ ] Validate all endpoints from catalog
- [ ] Test error handling
- [ ] Implement rate limiting
- [ ] Add retry logic
- [ ] Create normalization layer

---

## Key Findings

1. **API Base Path:** `/MobiControl/api`
2. **Authentication:** OAuth 2.0 password grant (not client credentials)
3. **Response Format:** Direct array (not wrapped in pagination object)
4. **Data Richness:** Very comprehensive device data available
5. **Device Count:** 27 devices in test environment

---

## Next Steps

1. **Test Additional Endpoints:**
   - `/MobiControl/api/devices/{deviceId}` - Single device
   - `/MobiControl/api/devices/{deviceId}/applications` - Applications
   - `/MobiControl/api/devices/{deviceId}/events` - Events
   - `/MobiControl/api/devices/{deviceId}/location` - Location

2. **Implement Normalization:**
   - Map MobiControl fields to canonical telemetry model
   - Handle nested objects (BatteryStatus, Memory, etc.)
   - Extract time-series data from device snapshots

3. **Update Documentation:**
   - Update API Integration Guide with actual endpoints
   - Document response formats
   - Add code examples

4. **Production Readiness:**
   - Add comprehensive error handling
   - Implement caching
   - Add monitoring and logging
   - Performance testing

---

## Security Notes

⚠️ **Important:** The test credentials are stored in `.env.test` which is gitignored.  
⚠️ **Never commit credentials to version control.**  
⚠️ **Rotate credentials regularly.**  
⚠️ **Use environment variables or secure secret management in production.**

---

**Last Updated:** December 2024

