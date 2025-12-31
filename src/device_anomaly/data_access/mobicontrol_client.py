"""SOTI MobiControl REST API Client.

This module provides a client for interacting with the SOTI MobiControl REST API.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from device_anomaly.config.settings import get_settings


class MobiControlAPIError(Exception):
    """Exception raised for MobiControl API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class MobiControlClient:
    """Client for SOTI MobiControl REST API."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        """Initialize MobiControl API client.

        Args:
            server_url: Base URL of MobiControl server (e.g., https://server.mobicontrol.cloud)
            client_id: OAuth client ID (if using OAuth)
            client_secret: OAuth client secret (if using OAuth)
            username: Username for basic auth (if using basic auth)
            password: Password for basic auth (if using basic auth)
            tenant_id: Tenant ID for multi-tenant deployments
        """
        settings = get_settings().mobicontrol

        self.server_url = (server_url or settings.server_url).rstrip("/")
        self.client_id = client_id or settings.client_id
        self.client_secret = client_secret or settings.client_secret
        self.username = username or settings.username
        self.password = password or settings.password
        self.tenant_id = tenant_id or settings.tenant_id

        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Authentication
        self.access_token: Optional[str] = None
        self.token_expiry: Optional[float] = None

        # Set up authentication
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with MobiControl API.

        Tries OAuth first, then falls back to basic auth if credentials are provided.
        """
        # Try OAuth if client credentials provided
        if self.client_id and self.client_secret:
            try:
                self._authenticate_oauth()
                return
            except Exception as e:
                print(f"OAuth authentication failed: {e}, trying basic auth...")

        # Fall back to basic auth if username/password provided
        if self.username and self.password:
            self._authenticate_basic()
            return

        raise ValueError("No authentication credentials provided")

    def _authenticate_oauth(self) -> None:
        """Authenticate using OAuth 2.0 password grant flow (MobiControl standard)."""
        # MobiControl uses password grant with /MobiControl/api/token
        token_endpoint = "/MobiControl/api/token"

        # Try password grant first (MobiControl standard)
        if self.username and self.password:
            token_data = {
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": self.client_id or "",
                "client_secret": self.client_secret or "",
            }

            try:
                url = f"{self.server_url}{token_endpoint}"
                response = self.session.post(
                    url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=10,
                )

                if response.status_code == 200:
                    token_response = response.json()
                    self.access_token = token_response.get("access_token")
                    expires_in = token_response.get("expires_in", 3600)
                    self.token_expiry = time.time() + expires_in - 60  # Refresh 1 min early
                    self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                    print(f"OAuth authentication successful via {token_endpoint}")
                    return
            except Exception as e:
                print(f"OAuth password grant failed: {e}")

        # Fallback to client credentials if no username/password
        if self.client_id and self.client_secret:
            token_data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }

            try:
                url = f"{self.server_url}{token_endpoint}"
                response = self.session.post(
                    url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=10,
                )

                if response.status_code == 200:
                    token_response = response.json()
                    self.access_token = token_response.get("access_token")
                    expires_in = token_response.get("expires_in", 3600)
                    self.token_expiry = time.time() + expires_in - 60
                    self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                    print(f"OAuth client credentials successful via {token_endpoint}")
                    return
            except Exception as e:
                print(f"OAuth client credentials failed: {e}")

        raise MobiControlAPIError("OAuth authentication failed")

    def _authenticate_basic(self) -> None:
        """Authenticate using basic authentication."""
        import base64

        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.session.headers["Authorization"] = f"Basic {encoded}"
        print("Basic authentication configured")

    def _ensure_authenticated(self) -> None:
        """Ensure authentication token is valid, refresh if needed."""
        if self.access_token and self.token_expiry:
            if time.time() >= self.token_expiry:
                self._authenticate_oauth()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request to MobiControl API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/api/v1/devices")
            params: Query parameters
            json_data: JSON body for POST/PUT requests
            **kwargs: Additional arguments for requests

        Returns:
            JSON response as dictionary

        Raises:
            MobiControlAPIError: If request fails
        """
        self._ensure_authenticated()

        # Add tenant context if provided
        if self.tenant_id:
            if params is None:
                params = {}
            params.setdefault("tenantId", self.tenant_id)
            self.session.headers.setdefault("X-Tenant-ID", self.tenant_id)

        url = f"{self.server_url}{endpoint}"
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            timeout=30,
            **kwargs,
        )

        # Handle errors
        if response.status_code >= 400:
            error_msg = f"API request failed: {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", error_msg)
            except Exception:
                error_msg = response.text or error_msg

            raise MobiControlAPIError(
                error_msg,
                status_code=response.status_code,
                response=response.json() if response.headers.get("content-type", "").startswith("application/json") else None,
            )

        # Return JSON response
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return {"data": response.text}

    def get_devices(
        self,
        page: int = 1,
        page_size: int = 100,
        filter_expr: Optional[str] = None,
        site_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get paginated list of devices.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            filter_expr: OData filter expression
            site_id: Site ID filter

        Returns:
            Dictionary with device data (structure depends on API version)
        """
        # MobiControl standard endpoint
        endpoint = "/MobiControl/api/devices"

        params = {}
        # Add pagination if supported
        if page > 1:
            params["page"] = page
        if page_size != 100:
            params["pageSize"] = page_size

        if filter_expr:
            params["filter"] = filter_expr
        if site_id:
            params["siteId"] = site_id

        return self._request("GET", endpoint, params=params)

    def get_device(self, device_id: str) -> Dict[str, Any]:
        """Get single device details.

        Args:
            device_id: Device ID

        Returns:
            Device details dictionary
        """
        endpoint = f"/MobiControl/api/devices/{device_id}"
        return self._request("GET", endpoint)

    def get_device_applications(self, device_id: str) -> List[Dict[str, Any]]:
        """Get installed applications for a device.

        Args:
            device_id: Device ID

        Returns:
            List of application dictionaries
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/applications"
        try:
            response = self._request("GET", endpoint)
            return response.get("data", response if isinstance(response, list) else [])
        except MobiControlAPIError as e:
            if e.status_code == 404:
                return []  # Endpoint not available
            raise

    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and return server information.

        Returns:
            Dictionary with connection test results
        """
        result = {
            "server_url": self.server_url,
            "authenticated": False,
            "endpoints_tested": [],
            "discovery_endpoints": [],
            "errors": [],
        }

        # Test authentication
        try:
            self._ensure_authenticated()
            result["authenticated"] = True
        except Exception as e:
            result["errors"].append(f"Authentication failed: {e}")
            return result

        # Try discovery endpoints first
        discovery_endpoints = [
            "/",
            "/api",
            "/api/v1",
            "/RestAPI",
            "/MobiControl/api",
            "/MobiControl/RestAPI",
            "/Services/RestAPI",
        ]

        for endpoint in discovery_endpoints:
            try:
                response = self.session.get(f"{self.server_url}{endpoint}", timeout=5)
                result["discovery_endpoints"].append(
                    {
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", ""),
                        "content_preview": response.text[:200] if response.text else "",
                    }
                )
            except Exception as e:
                result["discovery_endpoints"].append(
                    {
                        "endpoint": endpoint,
                        "status": "error",
                        "error": str(e),
                    }
                )

        # Test device endpoints with various paths
        test_endpoints = [
            "/api/v1/devices",
            "/api/devices",
            "/devices",
            "/RestAPI/Devices",
            "/MobiControl/api/v1/devices",
            "/MobiControl/RestAPI/Devices",
            "/Services/RestAPI/Devices",
            "/RestAPI/v1/Devices",
            "/RestAPI/v1/Device",
            "/api/v1/Device",
            "/RestAPI/Device",
        ]

        for endpoint in test_endpoints:
            try:
                response = self._request("GET", endpoint, params={"pageSize": 1})
                result["endpoints_tested"].append(
                    {
                        "endpoint": endpoint,
                        "status": "success",
                        "response_keys": list(response.keys()) if isinstance(response, dict) else "non-dict",
                    }
                )
            except Exception as e:
                error_msg = str(e)
                # Truncate long error messages
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                result["endpoints_tested"].append(
                    {
                        "endpoint": endpoint,
                        "status": "failed",
                        "error": error_msg,
                    }
                )

        return result

    # =========================================================================
    # Device Control Actions (SOTI MobiControl Remote Actions)
    # =========================================================================

    def lock_device(self, device_id: str) -> Dict[str, Any]:
        """Lock a device remotely.

        Args:
            device_id: Device ID to lock

        Returns:
            Dictionary with action result
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/actions/lock"
        return self._request("POST", endpoint)

    def restart_device(self, device_id: str) -> Dict[str, Any]:
        """Restart a device remotely.

        Args:
            device_id: Device ID to restart

        Returns:
            Dictionary with action result
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/actions/restart"
        return self._request("POST", endpoint)

    def wipe_device(self, device_id: str, factory_reset: bool = False) -> Dict[str, Any]:
        """Wipe a device remotely (enterprise or factory reset).

        Args:
            device_id: Device ID to wipe
            factory_reset: If True, perform factory reset; otherwise enterprise wipe

        Returns:
            Dictionary with action result
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/actions/wipe"
        return self._request("POST", endpoint, json_data={"factoryReset": factory_reset})

    def send_message(
        self,
        device_id: str,
        message: str,
        title: str = "Message from Admin",
    ) -> Dict[str, Any]:
        """Send a message to a device.

        Args:
            device_id: Device ID
            message: Message content
            title: Message title

        Returns:
            Dictionary with action result
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/actions/message"
        return self._request(
            "POST",
            endpoint,
            json_data={"title": title, "message": message},
        )

    def locate_device(self, device_id: str) -> Dict[str, Any]:
        """Request current location of a device.

        Args:
            device_id: Device ID

        Returns:
            Dictionary with location data or action acknowledgment
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/actions/locate"
        return self._request("POST", endpoint)

    def sync_device(self, device_id: str) -> Dict[str, Any]:
        """Force sync/check-in for a device.

        Args:
            device_id: Device ID

        Returns:
            Dictionary with action result
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/actions/checkin"
        return self._request("POST", endpoint)

    def clear_passcode(self, device_id: str) -> Dict[str, Any]:
        """Clear passcode on a device.

        Args:
            device_id: Device ID

        Returns:
            Dictionary with action result
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/actions/clearpasscode"
        return self._request("POST", endpoint)

    def clear_app_data(self, device_id: str, package_name: str) -> Dict[str, Any]:
        """Clear data for a specific app on the device.

        Args:
            device_id: Device ID
            package_name: Package name of the app to clear

        Returns:
            Dictionary with action result
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/actions/clearappdata"
        return self._request("POST", endpoint, json_data={"packageName": package_name})

    def get_device_location_history(
        self,
        device_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get location history for a device.

        Args:
            device_id: Device ID
            start_date: Start date for history
            end_date: End date for history

        Returns:
            List of location records
        """
        endpoint = f"/MobiControl/api/devices/{device_id}/locations"
        params = {}
        if start_date:
            params["startDate"] = start_date.isoformat()
        if end_date:
            params["endDate"] = end_date.isoformat()

        try:
            response = self._request("GET", endpoint, params=params)
            return response.get("data", response if isinstance(response, list) else [])
        except MobiControlAPIError as e:
            if e.status_code == 404:
                return []
            raise

