# SOTI XSight & MobiControl API Integration Guide

**Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Draft - Requires validation with actual API endpoints

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [MobiControl REST API](#mobicontrol-rest-api)
4. [XSight API (if available)](#xsight-api-if-available)
5. [Common Patterns](#common-patterns)
6. [Error Handling](#error-handling)
7. [Rate Limiting & Best Practices](#rate-limiting--best-practices)
8. [Sample Code](#sample-code)
9. [Testing & Validation](#testing--validation)

---

## Overview

This guide provides practical integration patterns for accessing device telemetry and management data from SOTI XSight and MobiControl via their REST APIs.

### Key Differences

| Aspect | MobiControl | XSight |
|--------|-------------|--------|
| **API Maturity** | REST API documented | API availability TBD |
| **Primary Use** | Device management, inventory, compliance | Real-time analytics, performance metrics |
| **Data Model** | Device-centric, policy-driven | Time-series, metric-driven |
| **Update Frequency** | Periodic (check-ins) | Near real-time |

---

## Authentication

### MobiControl REST API Authentication

**Note:** Authentication method may vary by version. Common approaches:

#### Option 1: API Key Authentication (if supported)

```http
GET /api/v1/devices
Authorization: Bearer <api_key>
X-API-Key: <api_key>
```

#### Option 2: OAuth 2.0 (if supported)

```http
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
&client_id=<client_id>
&client_secret=<client_secret>
&scope=device.read
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "device.read"
}
```

#### Option 3: Basic Authentication (legacy)

```http
GET /api/v1/devices
Authorization: Basic <base64(username:password)>
```

### XSight API Authentication

**Status:** TBD - Requires validation with SOTI documentation

**Assumed approach (to be validated):**
- Similar OAuth 2.0 or API key approach
- May require separate credentials from MobiControl

### Multi-Tenant Considerations

For multi-tenant deployments, include tenant context:

```http
GET /api/v1/devices
Authorization: Bearer <token>
X-Tenant-ID: <tenant_id>
X-Site-ID: <site_id>  # Optional, for site-scoped queries
```

---

## MobiControl REST API

### Base URL

```
https://<mobicontrol-server>/api/v1
```

**Note:** Version and base path may vary. Confirm with SOTI documentation.

### Core Endpoints

#### 1. Device Inventory

**Get All Devices**
```http
GET /devices
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `pageSize`: Items per page (default: 50, max: 1000)
- `filter`: Filter expression (e.g., `DeviceModel eq 'TC52'`)
- `orderBy`: Sort field (e.g., `DeviceName asc`)
- `tenantId`: Tenant filter (if multi-tenant)
- `siteId`: Site filter (optional)

**Response:**
```json
{
  "data": [
    {
      "deviceId": "12345",
      "deviceName": "Warehouse-Device-01",
      "serialNumber": "SN123456789",
      "imei": "123456789012345",
      "manufacturer": "Zebra",
      "model": "TC52",
      "osVersion": "Android 11",
      "securityPatchLevel": "2024-01-05",
      "batteryLevel": 85,
      "lastConnectedTime": "2024-12-24T10:30:00Z",
      "complianceStatus": "Compliant",
      "encryptionStatus": "Encrypted",
      "passcodeStatus": "Set",
      "jailbreakStatus": "Not Jailbroken",
      "tenantId": "tenant-001",
      "siteId": "site-001"
    }
  ],
  "pagination": {
    "page": 1,
    "pageSize": 50,
    "totalItems": 1250,
    "totalPages": 25
  }
}
```

**Get Single Device**
```http
GET /devices/{deviceId}
```

**Response:**
```json
{
  "deviceId": "12345",
  "deviceName": "Warehouse-Device-01",
  "serialNumber": "SN123456789",
  "imei": "123456789012345",
  "manufacturer": "Zebra",
  "model": "TC52",
  "deviceFamily": "Rugged Handheld",
  "processorType": "Snapdragon 660",
  "ramSize": "4096",
  "storageCapacity": "32768",
  "osVersion": "Android 11",
  "osBuildNumber": "RQ3A.210905.001",
  "securityPatchLevel": "2024-01-05",
  "androidSecurityPatchDate": "2024-01-05",
  "batteryLevel": 85,
  "chargingStatus": "Not Charging",
  "wifiSsid": "Warehouse-WiFi",
  "wifiSignalStrength": -65,
  "cellularCarrier": "Verizon",
  "cellularSignalStrength": -75,
  "cellularNetworkType": "LTE",
  "ipAddress": "192.168.1.100",
  "vpnStatus": "Connected",
  "lastConnectedTime": "2024-12-24T10:30:00Z",
  "uptime": 86400,
  "complianceStatus": "Compliant",
  "encryptionStatus": "Encrypted",
  "passcodeStatus": "Set",
  "passcodeComplexity": "Strong",
  "jailbreakStatus": "Not Jailbroken",
  "developerMode": "Disabled",
  "usbDebugging": "Disabled",
  "screenLockType": "PIN",
  "appliedPolicies": ["Security-Policy-001", "WiFi-Policy-002"],
  "customAttributes": {
    "department": "Warehouse",
    "assignedUser": "John Doe"
  }
}
```

#### 2. Application Inventory

**Get Installed Applications**
```http
GET /devices/{deviceId}/applications
```

**Response:**
```json
{
  "data": [
    {
      "packageName": "com.example.app",
      "appName": "Example App",
      "version": "1.2.3",
      "versionCode": 123,
      "installDate": "2024-01-15T08:00:00Z",
      "appSize": 52428800,
      "isSystemApp": false
    }
  ]
}
```

#### 3. Device Events & Logs

**Get Device Events**
```http
GET /devices/{deviceId}/events
```

**Query Parameters:**
- `startDate`: ISO 8601 datetime
- `endDate`: ISO 8601 datetime
- `eventType`: Filter by event type (e.g., `Reboot`, `AppInstall`, `PolicyApplied`)
- `page`, `pageSize`: Pagination

**Response:**
```json
{
  "data": [
    {
      "eventId": "evt-001",
      "deviceId": "12345",
      "eventType": "Reboot",
      "timestamp": "2024-12-24T08:00:00Z",
      "details": {
        "reason": "User initiated",
        "uptimeBeforeReboot": 172800
      }
    },
    {
      "eventId": "evt-002",
      "deviceId": "12345",
      "eventType": "AppInstall",
      "timestamp": "2024-12-24T09:15:00Z",
      "details": {
        "packageName": "com.example.app",
        "version": "1.2.3"
      }
    }
  ],
  "pagination": {
    "page": 1,
    "pageSize": 50,
    "totalItems": 150
  }
}
```

**Get Remote Control Sessions**
```http
GET /devices/{deviceId}/remote-control-sessions
```

**Response:**
```json
{
  "data": [
    {
      "sessionId": "rc-001",
      "deviceId": "12345",
      "initiatedBy": "admin@example.com",
      "startTime": "2024-12-24T10:00:00Z",
      "endTime": "2024-12-24T10:15:00Z",
      "duration": 900,
      "reason": "Troubleshooting"
    }
  ]
}
```

#### 4. Compliance & Policies

**Get Device Compliance Status**
```http
GET /devices/{deviceId}/compliance
```

**Response:**
```json
{
  "deviceId": "12345",
  "overallStatus": "Compliant",
  "policies": [
    {
      "policyId": "Security-Policy-001",
      "policyName": "Encryption Required",
      "status": "Compliant",
      "lastChecked": "2024-12-24T10:30:00Z"
    },
    {
      "policyId": "WiFi-Policy-002",
      "policyName": "Authorized Networks Only",
      "status": "Non-Compliant",
      "lastChecked": "2024-12-24T10:30:00Z",
      "violations": [
        {
          "violationType": "UnauthorizedNetwork",
          "details": "Connected to 'Public-WiFi'"
        }
      ]
    }
  ]
}
```

#### 5. Location & Geofencing

**Get Device Location**
```http
GET /devices/{deviceId}/location
```

**Response:**
```json
{
  "deviceId": "12345",
  "latitude": 40.7128,
  "longitude": -74.0060,
  "accuracy": 10.5,
  "timestamp": "2024-12-24T10:30:00Z",
  "geofenceStatus": [
    {
      "geofenceId": "gf-001",
      "geofenceName": "Warehouse Zone A",
      "status": "Inside"
    }
  ]
}
```

**Get Location History**
```http
GET /devices/{deviceId}/location/history
```

**Query Parameters:**
- `startDate`, `endDate`: ISO 8601 datetime range
- `page`, `pageSize`: Pagination

#### 6. Peripherals

**Get Connected Peripherals**
```http
GET /devices/{deviceId}/peripherals
```

**Response:**
```json
{
  "data": [
    {
      "peripheralType": "BarcodeScanner",
      "peripheralName": "Zebra SE4710",
      "status": "Connected",
      "lastSeen": "2024-12-24T10:30:00Z"
    },
    {
      "peripheralType": "Bluetooth",
      "peripheralName": "BT-Headset-01",
      "status": "Paired",
      "lastSeen": "2024-12-24T10:25:00Z"
    }
  ]
}
```

#### 7. Custom Attributes

**Get Custom Attributes**
```http
GET /devices/{deviceId}/custom-attributes
```

**Response:**
```json
{
  "deviceId": "12345",
  "attributes": {
    "department": "Warehouse",
    "assignedUser": "John Doe",
    "costCenter": "CC-001",
    "warrantyExpiry": "2025-12-31"
  }
}
```

---

## XSight API (if available)

**Status:** API availability and endpoints require validation with SOTI documentation.

### Assumed Endpoints (to be validated)

#### 1. Real-Time Device Metrics

**Get Device Performance Metrics**
```http
GET /api/v1/devices/{deviceId}/metrics
```

**Query Parameters:**
- `startTime`: ISO 8601 datetime
- `endTime`: ISO 8601 datetime
- `metrics`: Comma-separated list (e.g., `cpu,memory,storage,temperature`)

**Response (assumed):**
```json
{
  "deviceId": "12345",
  "timestamp": "2024-12-24T10:30:00Z",
  "metrics": {
    "cpuUsage": 45.2,
    "memoryUsage": 2048,
    "memoryAvailable": 2048,
    "storageUsage": 16384,
    "storageAvailable": 16384,
    "deviceTemperature": 35.5,
    "cpuTemperature": 42.0,
    "cpuFrequency": 1800
  }
}
```

#### 2. Battery Metrics

**Get Battery Health**
```http
GET /api/v1/devices/{deviceId}/battery
```

**Response (assumed):**
```json
{
  "deviceId": "12345",
  "batteryLevel": 85,
  "batteryHealth": "Good",
  "chargingStatus": "Not Charging",
  "batteryTemperature": 28.5,
  "powerSource": "Battery",
  "timestamp": "2024-12-24T10:30:00Z"
}
```

#### 3. Network Metrics

**Get Network Status**
```http
GET /api/v1/devices/{deviceId}/network
```

**Response (assumed):**
```json
{
  "deviceId": "12345",
  "cellularSignalStrength": -75,
  "wifiSignalStrength": -65,
  "networkType": "LTE",
  "timestamp": "2024-12-24T10:30:00Z"
}
```

#### 4. Running Applications

**Get Running Applications**
```http
GET /api/v1/devices/{deviceId}/running-apps
```

**Response (assumed):**
```json
{
  "deviceId": "12345",
  "timestamp": "2024-12-24T10:30:00Z",
  "runningApps": [
    {
      "packageName": "com.example.app",
      "appName": "Example App",
      "cpuUsage": 15.5,
      "memoryUsage": 256
    }
  ]
}
```

#### 5. Alerts

**Get Device Alerts**
```http
GET /api/v1/devices/{deviceId}/alerts
```

**Query Parameters:**
- `startTime`, `endTime`: ISO 8601 datetime range
- `severity`: Filter by severity (Critical, Warning, Info)
- `page`, `pageSize`: Pagination

**Response (assumed):**
```json
{
  "data": [
    {
      "alertId": "alert-001",
      "deviceId": "12345",
      "severity": "Warning",
      "alertType": "HighCPUUsage",
      "message": "CPU usage above 80% for 5 minutes",
      "timestamp": "2024-12-24T10:25:00Z",
      "resolved": false
    }
  ]
}
```

---

## Common Patterns

### 1. Pagination Pattern

```python
def fetch_all_devices(base_url, headers, max_pages=None):
    """Fetch all devices using pagination."""
    devices = []
    page = 1
    page_size = 100
    
    while True:
        url = f"{base_url}/devices?page={page}&pageSize={page_size}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        devices.extend(data["data"])
        
        if page >= data["pagination"]["totalPages"]:
            break
        if max_pages and page >= max_pages:
            break
            
        page += 1
    
    return devices
```

### 2. Incremental Data Sync

```python
def sync_device_data_since(device_id, last_sync_time, base_url, headers):
    """Sync device data since last sync time."""
    # Get events since last sync
    events_url = (
        f"{base_url}/devices/{device_id}/events"
        f"?startDate={last_sync_time.isoformat()}"
    )
    events_response = requests.get(events_url, headers=headers)
    events_response.raise_for_status()
    events = events_response.json()["data"]
    
    # Get current device state
    device_url = f"{base_url}/devices/{device_id}"
    device_response = requests.get(device_url, headers=headers)
    device_response.raise_for_status()
    device = device_response.json()
    
    return {
        "device": device,
        "events": events,
        "syncTime": datetime.now(timezone.utc)
    }
```

### 3. Multi-Tenant Query Pattern

```python
def get_tenant_devices(tenant_id, site_id=None, base_url, headers):
    """Get devices for a specific tenant (and optionally site)."""
    params = {"tenantId": tenant_id}
    if site_id:
        params["siteId"] = site_id
    
    url = f"{base_url}/devices"
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    return response.json()["data"]
```

### 4. Batch Device Queries

```python
def get_multiple_devices(device_ids, base_url, headers):
    """Get multiple devices efficiently."""
    # Option 1: Use filter parameter (if supported)
    device_ids_str = ",".join(device_ids)
    url = f"{base_url}/devices?filter=DeviceId in ({device_ids_str})"
    
    # Option 2: Parallel requests (if filter not supported)
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = [
    #         executor.submit(
    #             requests.get,
    #             f"{base_url}/devices/{device_id}",
    #             headers=headers
    #         )
    #         for device_id in device_ids
    #     ]
    #     devices = [f.result().json() for f in futures]
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["data"]
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Handling |
|------|---------|----------|
| 200 | Success | Process response |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Validate request parameters |
| 401 | Unauthorized | Refresh authentication token |
| 403 | Forbidden | Check permissions/tenant access |
| 404 | Not Found | Device/resource doesn't exist |
| 429 | Too Many Requests | Implement exponential backoff |
| 500 | Server Error | Retry with backoff |
| 503 | Service Unavailable | Retry after delay |

### Retry Strategy

```python
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retry():
    """Create requests session with retry strategy."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session
```

### Error Response Format

```json
{
  "error": {
    "code": "DEVICE_NOT_FOUND",
    "message": "Device with ID 12345 not found",
    "details": {
      "deviceId": "12345",
      "tenantId": "tenant-001"
    }
  }
}
```

---

## Rate Limiting & Best Practices

### Rate Limits

**Note:** Actual rate limits need validation with SOTI documentation.

**Assumed limits:**
- MobiControl: 100 requests/minute per API key
- XSight: TBD

### Best Practices

1. **Use Pagination**
   - Don't request all data at once
   - Use appropriate page sizes (50-100 items)

2. **Implement Caching**
   - Cache device inventory (changes infrequently)
   - Cache authentication tokens
   - Use ETags if supported

3. **Batch Operations**
   - Group related queries when possible
   - Use filter parameters instead of multiple requests

4. **Respect Rate Limits**
   - Implement exponential backoff
   - Use request queuing/throttling
   - Monitor rate limit headers

5. **Efficient Polling**
   - Use incremental sync (query since last update)
   - Poll at appropriate intervals (not too frequent)
   - Use webhooks if available (preferred over polling)

### Rate Limit Headers (if supported)

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1703424000
```

---

## Sample Code

### Python Client Example

```python
import requests
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import time

class SOTIMobiControlClient:
    """Client for SOTI MobiControl REST API."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        oauth_token: Optional[str] = None,
        tenant_id: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.tenant_id = tenant_id
        self.session = requests.Session()
        
        # Set authentication
        if oauth_token:
            self.session.headers['Authorization'] = f'Bearer {oauth_token}'
        elif api_key:
            self.session.headers['X-API-Key'] = api_key
        
        # Set tenant context if provided
        if tenant_id:
            self.session.headers['X-Tenant-ID'] = tenant_id
    
    def get_devices(
        self,
        page: int = 1,
        page_size: int = 100,
        filter_expr: Optional[str] = None,
        site_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get paginated list of devices."""
        url = f"{self.base_url}/api/v1/devices"
        params = {
            "page": page,
            "pageSize": page_size
        }
        
        if filter_expr:
            params["filter"] = filter_expr
        if site_id:
            params["siteId"] = site_id
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_device(self, device_id: str) -> Dict[str, Any]:
        """Get single device details."""
        url = f"{self.base_url}/api/v1/devices/{device_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_device_applications(self, device_id: str) -> List[Dict[str, Any]]:
        """Get installed applications for a device."""
        url = f"{self.base_url}/api/v1/devices/{device_id}/applications"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json().get("data", [])
    
    def get_device_events(
        self,
        device_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get device events."""
        url = f"{self.base_url}/api/v1/devices/{device_id}/events"
        params = {}
        
        if start_date:
            params["startDate"] = start_date.isoformat()
        if end_date:
            params["endDate"] = end_date.isoformat()
        if event_type:
            params["eventType"] = event_type
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json().get("data", [])
    
    def get_device_location(self, device_id: str) -> Dict[str, Any]:
        """Get current device location."""
        url = f"{self.base_url}/api/v1/devices/{device_id}/location"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_all_devices(self, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch all devices using pagination."""
        all_devices = []
        page = 1
        
        while True:
            result = self.get_devices(page=page)
            all_devices.extend(result["data"])
            
            pagination = result["pagination"]
            if page >= pagination["totalPages"]:
                break
            if max_pages and page >= max_pages:
                break
            
            page += 1
            time.sleep(0.1)  # Small delay to respect rate limits
        
        return all_devices


# Usage example
if __name__ == "__main__":
    client = SOTIMobiControlClient(
        base_url="https://mobicontrol.example.com",
        api_key="your-api-key",
        tenant_id="tenant-001"
    )
    
    # Get all devices
    devices = client.get_all_devices()
    print(f"Found {len(devices)} devices")
    
    # Get specific device
    device = client.get_device("12345")
    print(f"Device: {device['deviceName']}")
    
    # Get device applications
    apps = client.get_device_applications("12345")
    print(f"Installed apps: {len(apps)}")
    
    # Get recent events
    from datetime import timedelta
    start_date = datetime.now(timezone.utc) - timedelta(days=7)
    events = client.get_device_events("12345", start_date=start_date)
    print(f"Events in last 7 days: {len(events)}")
```

### OAuth Token Refresh Example

```python
class SOTIClientWithOAuth(SOTIMobiControlClient):
    """Client with automatic OAuth token refresh."""
    
    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        scope: str = "device.read",
        tenant_id: Optional[str] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.token_expiry = None
        
        # Get initial token
        token = self._get_access_token(base_url)
        super().__init__(base_url, oauth_token=token, tenant_id=tenant_id)
    
    def _get_access_token(self, base_url: str) -> str:
        """Get OAuth access token."""
        url = f"{base_url}/oauth/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self.token_expiry = datetime.now(timezone.utc).timestamp() + token_data["expires_in"]
        
        return token_data["access_token"]
    
    def _ensure_valid_token(self):
        """Refresh token if expired."""
        if not self.token_expiry or datetime.now(timezone.utc).timestamp() >= self.token_expiry - 60:
            token = self._get_access_token(self.base_url)
            self.session.headers['Authorization'] = f'Bearer {token}'
    
    def get_device(self, device_id: str) -> Dict[str, Any]:
        """Get device with automatic token refresh."""
        self._ensure_valid_token()
        return super().get_device(device_id)
```

---

## Testing & Validation

### 1. API Endpoint Validation

**Checklist:**
- [ ] Verify base URL and API version
- [ ] Test authentication method
- [ ] Validate response formats
- [ ] Confirm pagination behavior
- [ ] Test error responses
- [ ] Verify rate limits

### 2. Data Validation

**Checklist:**
- [ ] Verify device identifier consistency (Device ID, Serial, IMEI)
- [ ] Confirm date/time formats (ISO 8601)
- [ ] Validate data types and null handling
- [ ] Check multi-tenant data isolation
- [ ] Verify custom attributes access

### 3. Integration Testing

```python
import pytest

def test_device_retrieval(client):
    """Test device retrieval."""
    devices = client.get_devices(page_size=10)
    assert "data" in devices
    assert "pagination" in devices
    assert len(devices["data"]) <= 10

def test_device_details(client):
    """Test device details retrieval."""
    # Get first device ID
    devices = client.get_devices(page_size=1)
    device_id = devices["data"][0]["deviceId"]
    
    # Get device details
    device = client.get_device(device_id)
    assert device["deviceId"] == device_id
    assert "deviceName" in device

def test_pagination(client):
    """Test pagination."""
    page1 = client.get_devices(page=1, page_size=10)
    page2 = client.get_devices(page=2, page_size=10)
    
    assert page1["data"] != page2["data"]
    assert len(page1["data"]) <= 10
```

### 4. Performance Testing

- Measure response times
- Test with large page sizes
- Validate rate limit handling
- Test concurrent requests

---

## Next Steps

1. **Obtain API Credentials**
   - Request API keys or OAuth credentials from SOTI
   - Confirm authentication method

2. **Validate Endpoints**
   - Test actual API endpoints
   - Confirm request/response formats
   - Document any differences from this guide

3. **Implement Client Library**
   - Create production-ready client
   - Add error handling and retries
   - Implement caching strategy

4. **Integration Testing**
   - Test with real devices
   - Validate data accuracy
   - Test multi-tenant scenarios

---

**Document Status:** Draft - Requires validation with actual SOTI API endpoints  
**Last Updated:** December 2024

