"""Device grouping service with AI-powered pattern analysis.

Groups devices affected by an insight using multiple strategies:
1. By location - geographic grouping
2. By device model - hardware grouping
3. By pattern similarity - AI-powered clustering based on behavior
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from device_anomaly.api.models import (
    DeviceGroupingResponse,
    ImpactedDeviceResponse,
)
from device_anomaly.llm.client import BaseLLMClient, get_default_llm_client

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Internal representation of device data for grouping."""

    device_id: int
    device_name: str | None
    device_model: str | None
    location: str | None
    status: str
    last_seen: str | None
    os_version: str | None
    anomaly_count: int = 0
    severity: str | None = None
    primary_metric: str | None = None


class DeviceGrouper:
    """Groups devices by various criteria including AI-powered pattern similarity."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        enable_ai_grouping: bool = True,
    ):
        """
        Initialize the device grouper.

        Args:
            llm_client: Optional LLM client for AI-powered grouping
            enable_ai_grouping: Whether to enable AI pattern analysis
        """
        self._llm_client = llm_client
        self._enable_ai_grouping = enable_ai_grouping

    def group_by_location(
        self, devices: list[ImpactedDeviceResponse]
    ) -> list[DeviceGroupingResponse]:
        """
        Group devices by their location.

        Args:
            devices: List of devices to group

        Returns:
            List of device groupings by location
        """
        groups: dict[str, list[ImpactedDeviceResponse]] = {}

        for device in devices:
            key = device.location or "Unknown Location"
            groups.setdefault(key, []).append(device)

        return [
            DeviceGroupingResponse(
                group_key=self._normalize_key(label),
                group_label=label,
                device_count=len(devs),
                devices=devs,
            )
            for label, devs in sorted(groups.items(), key=lambda x: -len(x[1]))
        ]

    def group_by_model(
        self, devices: list[ImpactedDeviceResponse]
    ) -> list[DeviceGroupingResponse]:
        """
        Group devices by their device model.

        Args:
            devices: List of devices to group

        Returns:
            List of device groupings by model
        """
        groups: dict[str, list[ImpactedDeviceResponse]] = {}

        for device in devices:
            key = device.device_model or "Unknown Model"
            groups.setdefault(key, []).append(device)

        return [
            DeviceGroupingResponse(
                group_key=self._normalize_key(label),
                group_label=label,
                device_count=len(devs),
                devices=devs,
            )
            for label, devs in sorted(groups.items(), key=lambda x: -len(x[1]))
        ]

    def group_by_pattern_similarity(
        self,
        devices: list[ImpactedDeviceResponse],
        insight_category: str | None = None,
        insight_headline: str | None = None,
    ) -> tuple[list[DeviceGroupingResponse], str | None]:
        """
        AI-powered grouping based on pattern similarity.

        Uses LLM to analyze device characteristics and identify
        meaningful clusters beyond simple attribute matching.

        Args:
            devices: List of devices to group
            insight_category: Category of the insight for context
            insight_headline: Headline of the insight for context

        Returns:
            Tuple of (groupings list, AI analysis text)
        """
        if not devices:
            return [], None

        if len(devices) < 3:
            # Not enough devices for meaningful pattern analysis
            return self._fallback_grouping(devices), "Insufficient devices for pattern analysis."

        if not self._enable_ai_grouping:
            return self._fallback_grouping(devices), "AI grouping is disabled."

        try:
            llm = self._llm_client or get_default_llm_client()
        except Exception as e:
            logger.warning(f"Failed to get LLM client: {e}")
            return self._fallback_grouping(devices), f"LLM unavailable: {str(e)}"

        # Build device summary for LLM (limit to 50 for context window)
        device_summaries = []
        for d in devices[:50]:
            severity_str = f", severity={d.severity}" if d.severity else ""
            metric_str = f", issue={d.primary_metric}" if d.primary_metric else ""
            device_summaries.append(
                f"- {d.device_name or f'Device-{d.device_id}'}: "
                f"model={d.device_model or 'Unknown'}, "
                f"location={d.location or 'Unknown'}, "
                f"status={d.status}"
                f"{severity_str}{metric_str}"
            )

        category_context = f' related to "{insight_category}"' if insight_category else ""
        headline_context = f"\nIssue: {insight_headline}" if insight_headline else ""

        prompt = f"""Analyze these {len(devices)} devices affected by an issue{category_context}.
{headline_context}

Devices:
{chr(10).join(device_summaries)}

Task: Identify 2-4 meaningful patterns or clusters among these devices that would help IT prioritize troubleshooting.

For each pattern, provide:
1. A brief, descriptive pattern name (2-5 words)
2. The device names that belong to this pattern (comma-separated)
3. A short explanation of why this grouping is meaningful

Format your response EXACTLY as follows (one pattern per block):
PATTERN: <pattern name>
DEVICES: <device1>, <device2>, ...
REASON: <1 sentence explanation>

Focus on actionable groupings based on:
- Common hardware characteristics
- Geographic proximity
- Similar issue severity
- Shared infrastructure (same charging stations, network segments, etc.)
- Usage patterns (if inferable from names/locations)"""

        try:
            analysis = llm.generate(prompt, temperature=0.3, max_tokens=800)
            groupings = self._parse_pattern_response(analysis, devices)

            if groupings:
                # Create a summary from the analysis
                summary = self._extract_analysis_summary(analysis, len(devices))
                return groupings, summary
            else:
                # Parsing failed, use fallback
                logger.warning("Failed to parse LLM response, using fallback grouping")
                return self._fallback_grouping(devices), "Pattern analysis parsing failed. Using fallback grouping."

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._fallback_grouping(devices), f"Pattern analysis unavailable: {str(e)}"

    def _parse_pattern_response(
        self,
        response: str,
        devices: list[ImpactedDeviceResponse],
    ) -> list[DeviceGroupingResponse]:
        """
        Parse LLM response into device groupings.

        Args:
            response: LLM response text
            devices: Original device list for mapping

        Returns:
            List of device groupings
        """
        groupings: list[DeviceGroupingResponse] = []

        # Build device name lookup (case-insensitive)
        device_by_name: dict[str, ImpactedDeviceResponse] = {}
        for d in devices:
            name = (d.device_name or f"Device-{d.device_id}").lower().strip()
            device_by_name[name] = d
            # Also add without common prefixes/suffixes for fuzzy matching
            clean_name = re.sub(r'^(device[-_]?|dev[-_]?)', '', name, flags=re.IGNORECASE)
            if clean_name:
                device_by_name[clean_name] = d

        # Parse PATTERN/DEVICES/REASON blocks
        pattern_blocks = re.split(r'\n(?=PATTERN:)', response, flags=re.IGNORECASE)

        for block in pattern_blocks:
            if not block.strip():
                continue

            # Extract pattern name
            pattern_match = re.search(r'PATTERN:\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
            if not pattern_match:
                continue
            pattern_name = pattern_match.group(1).strip()

            # Extract device list
            devices_match = re.search(r'DEVICES:\s*(.+?)(?:\nREASON:|$)', block, re.IGNORECASE | re.DOTALL)
            if not devices_match:
                continue

            device_names_str = devices_match.group(1).strip()
            # Split by comma, handling various formats
            device_names = [n.strip() for n in re.split(r',\s*|\n', device_names_str) if n.strip()]

            # Map names to actual devices
            matched_devices: list[ImpactedDeviceResponse] = []
            for name in device_names:
                name_lower = name.lower().strip()
                # Try exact match first
                if name_lower in device_by_name:
                    matched_devices.append(device_by_name[name_lower])
                else:
                    # Try partial match
                    for key, device in device_by_name.items():
                        if name_lower in key or key in name_lower:
                            if device not in matched_devices:
                                matched_devices.append(device)
                            break

            if matched_devices:
                groupings.append(
                    DeviceGroupingResponse(
                        group_key=self._normalize_key(pattern_name),
                        group_label=pattern_name,
                        device_count=len(matched_devices),
                        devices=matched_devices,
                    )
                )

        return groupings

    def _extract_analysis_summary(self, analysis: str, total_devices: int) -> str:
        """Extract a summary from the LLM analysis."""
        # Count patterns found
        pattern_count = len(re.findall(r'PATTERN:', analysis, re.IGNORECASE))

        # Extract reasons for a brief summary
        reasons = re.findall(r'REASON:\s*(.+?)(?:\n|$)', analysis, re.IGNORECASE)
        reason_summary = " ".join(reasons[:2]) if reasons else ""

        if pattern_count > 0:
            summary = f"Identified {pattern_count} patterns across {total_devices} devices."
            if reason_summary:
                summary += f" Key findings: {reason_summary}"
            return summary
        return "Pattern analysis completed."

    def _fallback_grouping(
        self, devices: list[ImpactedDeviceResponse]
    ) -> list[DeviceGroupingResponse]:
        """
        Combined location + model grouping as fallback when AI is unavailable.

        Args:
            devices: List of devices to group

        Returns:
            List of device groupings
        """
        groups: dict[str, list[ImpactedDeviceResponse]] = {}

        for device in devices:
            location = device.location or "Unknown Location"
            model = device.device_model or "Unknown Model"
            key = f"{location} / {model}"
            groups.setdefault(key, []).append(device)

        return [
            DeviceGroupingResponse(
                group_key=self._normalize_key(label),
                group_label=label,
                device_count=len(devs),
                devices=devs,
            )
            for label, devs in sorted(groups.items(), key=lambda x: -len(x[1]))
        ]

    def _normalize_key(self, label: str) -> str:
        """Normalize a label into a URL-safe key."""
        return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
