"""Test script for SOTI MobiControl API integration.

This script tests the MobiControl API connection and validates endpoints.
"""

from __future__ import annotations

# Prevent pytest from collecting this CLI utility as a test module.
__test__ = False

import json
import os
import sys

# Load test credentials from .env.test if it exists
try:
    from dotenv import load_dotenv

    env_file = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env.test")
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    pass

from device_anomaly.data_access.mobicontrol_client import MobiControlAPIError, MobiControlClient


def test_connection():
    """Test basic API connection."""
    print("=" * 80)
    print("Testing MobiControl API Connection")
    print("=" * 80)

    try:
        client = MobiControlClient()
        result = client.test_connection()

        print("\nConnection Test Results:")
        print(f"Server URL: {result['server_url']}")
        print(f"Authenticated: {result['authenticated']}")

        if result.get("discovery_endpoints"):
            print(f"\nDiscovery Endpoints Tested: {len(result['discovery_endpoints'])}")
            for disc in result["discovery_endpoints"]:
                print(f"\n  {disc['endpoint']}:")
                if "status_code" in disc:
                    print(f"    Status Code: {disc['status_code']}")
                    print(f"    Content Type: {disc.get('content_type', 'N/A')}")
                    if disc.get("content_preview"):
                        print(f"    Content Preview: {disc['content_preview'][:100]}...")
                else:
                    print(f"    Error: {disc.get('error', 'N/A')}")

        print(f"\nDevice Endpoints Tested: {len(result['endpoints_tested'])}")
        for endpoint_test in result["endpoints_tested"]:
            print(f"\n  {endpoint_test['endpoint']}:")
            print(f"    Status: {endpoint_test['status']}")
            if endpoint_test["status"] == "success":
                print(f"    Response keys: {endpoint_test.get('response_keys', 'N/A')}")
            else:
                error = endpoint_test.get("error", "N/A")
                if len(error) > 150:
                    error = error[:150] + "..."
                print(f"    Error: {error}")

        if result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"  - {error}")

        return result["authenticated"] and any(
            et["status"] == "success" for et in result["endpoints_tested"]
        )

    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_get_devices():
    """Test getting device list."""
    print("\n" + "=" * 80)
    print("Testing Get Devices Endpoint")
    print("=" * 80)

    try:
        client = MobiControlClient()

        print("\nFetching first page of devices (page_size=5)...")
        result = client.get_devices(page=1, page_size=5)

        print("\n✅ Successfully retrieved devices")
        print(f"Response type: {type(result).__name__}")

        # Handle both list and dict responses
        if isinstance(result, list):
            devices = result
            print(f"\n  Number of devices: {len(devices)}")

            if devices:
                print("\n  First device sample:")
                first_device = devices[0]
                if isinstance(first_device, dict):
                    print(f"    Keys: {list(first_device.keys())}")
                    print("    Sample data:")
                    for key in list(first_device.keys())[:15]:  # Show first 15 keys
                        value = first_device[key]
                        if isinstance(value, (dict, list)):
                            print(f"      {key}: {type(value).__name__} ({len(value) if hasattr(value, '__len__') else 'N/A'})")
                        else:
                            print(f"      {key}: {value}")
                else:
                    print(f"    Device data: {first_device}")

        elif isinstance(result, dict):
            print("Response structure:")
            print(f"  Keys: {list(result.keys())}")

            if "data" in result:
                devices = result["data"]
                print(f"\n  Number of devices: {len(devices)}")

                if devices:
                    print("\n  First device sample:")
                    first_device = devices[0]
                    print(f"    Keys: {list(first_device.keys())}")
                    print("    Sample data:")
                    for key in list(first_device.keys())[:15]:  # Show first 15 keys
                        value = first_device[key]
                        if isinstance(value, (dict, list)):
                            print(f"      {key}: {type(value).__name__} ({len(value) if hasattr(value, '__len__') else 'N/A'})")
                        else:
                            print(f"      {key}: {value}")

            if "pagination" in result:
                pagination = result["pagination"]
                print("\n  Pagination info:")
                for key, value in pagination.items():
                    print(f"    {key}: {value}")
        else:
            print(f"  Unexpected response type: {type(result)}")
            print(f"  Response: {result}")

        return True

    except MobiControlAPIError as e:
        print(f"\n❌ API Error: {e}")
        if e.status_code:
            print(f"  Status code: {e.status_code}")
        if e.response:
            print(f"  Response: {json.dumps(e.response, indent=2)}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_get_single_device():
    """Test getting a single device."""
    print("\n" + "=" * 80)
    print("Testing Get Single Device Endpoint")
    print("=" * 80)

    try:
        client = MobiControlClient()

        # First, get a device ID from the list
        print("\nFetching device list to get a device ID...")
        devices_result = client.get_devices(page=1, page_size=1)

        # Handle both list and dict responses
        devices_list = devices_result if isinstance(devices_result, list) else devices_result.get("data", [])

        if not devices_list:
            print("❌ No devices found to test with")
            return False

        first_device = devices_list[0]
        if not isinstance(first_device, dict):
            print(f"❌ Unexpected device data format: {type(first_device)}")
            return False

        device_id = first_device.get("deviceId") or first_device.get("id") or first_device.get("DeviceId")
        if not device_id:
            # Try to find any ID-like field
            for key in first_device:
                if "id" in key.lower() and key.lower() != "guid":
                    device_id = first_device[key]
                    break

        if not device_id:
            print("❌ Could not find device ID in response")
            print(f"   Available keys: {list(first_device.keys())}")
            return False

        print(f"\nFetching details for device: {device_id}")
        device = client.get_device(str(device_id))

        print("\n✅ Successfully retrieved device details")
        print(f"  Device keys: {list(device.keys())}")
        print("\n  Sample device data:")
        for key in list(device.keys())[:15]:  # Show first 15 keys
            value = device[key]
            if isinstance(value, (dict, list)):
                print(f"    {key}: {type(value).__name__} ({len(value) if hasattr(value, '__len__') else 'N/A'})")
            else:
                print(f"    {key}: {value}")

        return True

    except MobiControlAPIError as e:
        print(f"\n❌ API Error: {e}")
        if e.status_code:
            print(f"  Status code: {e.status_code}")
        if e.response:
            print(f"  Response: {json.dumps(e.response, indent=2)}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SOTI MobiControl API Integration Test")
    print("=" * 80)

    results = []

    # Test 1: Connection
    results.append(("Connection Test", test_connection()))

    # Test 2: Get Devices
    results.append(("Get Devices", test_get_devices()))

    # Test 3: Get Single Device
    results.append(("Get Single Device", test_get_single_device()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
