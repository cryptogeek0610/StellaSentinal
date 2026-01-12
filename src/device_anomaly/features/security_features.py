"""
Security Posture Feature Engineering for Anomaly Detection.

This module computes comprehensive security features including:
- Security compliance scores
- Encryption and passcode status
- Root/jailbreak detection
- Security patch age
- Risk indicators (USB debugging, developer mode)

Data Sources:
- MobiControl: DevInfo security columns
- Policy compliance states
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Security indicator weights for composite scoring
SECURITY_INDICATORS: Dict[str, Tuple[float, bool]] = {
    # (weight, is_positive)
    # Positive indicators (having these is good)
    "HasPasscode": (1.0, True),
    "IsEncrypted": (1.0, True),
    "IsAndroidSafetynetAttestationPassed": (1.0, True),
    "IsSupervised": (0.5, True),  # iOS supervised mode
    "FileVaultEnabled": (1.0, True),  # Mac encryption
    "IsSystemIntegrityProtectionEnabled": (1.0, True),  # Mac SIP
    "KnoxAttestationStatus": (0.5, True),  # Samsung Knox
    "TrustStatus": (0.5, True),
    # Negative indicators (having these is bad)
    "IsRooted": (1.0, False),
    "IsJailbroken": (1.0, False),
    "IsDeveloperModeEnabled": (0.5, False),
    "IsUSBDebuggingEnabled": (0.5, False),
    "CompromisedStatus": (1.0, False),
    "IsDeviceCompromised": (1.0, False),
    # Policy compliance
    "IsCompliant": (0.5, True),
    "PolicyCompliant": (0.5, True),
}


class SecurityFeatureBuilder:
    """
    Security posture feature engineering for device anomaly detection.

    Computes comprehensive security metrics:
    - Composite security score (0-1)
    - Risk categorization (low, medium, high, critical)
    - Individual security indicator flags
    - Patch age and compliance metrics
    """

    def __init__(
        self,
        patch_age_warning_days: int = 90,
        patch_age_critical_days: int = 180,
    ):
        """
        Initialize the security feature builder.

        Args:
            patch_age_warning_days: Days without security patch before warning
            patch_age_critical_days: Days without security patch before critical
        """
        self.patch_age_warning_days = patch_age_warning_days
        self.patch_age_critical_days = patch_age_critical_days

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build security posture features.

        Expects columns from MobiControl DevInfo like:
          - HasPasscode, IsEncrypted, IsRooted, IsJailbroken
          - ComplianceState, SecurityPatchLevel
          - IsDeveloperModeEnabled, IsUSBDebuggingEnabled

        Returns DataFrame with security features added.
        """
        if df.empty:
            return df

        df = df.copy()

        # Compute composite security score
        logger.info("Computing composite security score...")
        df = self._add_composite_security_score(df)

        # Add individual security flags
        logger.info("Computing individual security flags...")
        df = self._add_security_flags(df)

        # Add patch age features
        logger.info("Computing security patch age features...")
        df = self._add_patch_age_features(df)

        # Add risk categorization
        logger.info("Computing risk categories...")
        df = self._add_risk_categories(df)

        # Add compliance features
        logger.info("Computing compliance features...")
        df = self._add_compliance_features(df)

        return df

    def _add_composite_security_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute weighted composite security score (0-1)."""
        # Initialize score and weights
        score = pd.Series(0.0, index=df.index)
        max_positive_weight = 0.0
        max_negative_weight = 0.0

        for indicator, (weight, is_positive) in SECURITY_INDICATORS.items():
            if indicator not in df.columns:
                continue

            # Convert to numeric boolean
            val = df[indicator].fillna(False)
            if val.dtype == "object":
                val = val.map(lambda x: str(x).lower() in ("true", "1", "yes", "passed"))
            val = val.astype(float)

            if is_positive:
                # Having this indicator is good
                score += val * weight
                max_positive_weight += weight
            else:
                # Having this indicator is bad
                score -= val * weight
                max_negative_weight += weight

        # Normalize to 0-1 scale
        total_weight = max_positive_weight + max_negative_weight
        if total_weight > 0:
            # Shift to positive range and normalize
            df["security_score"] = ((score + max_negative_weight) / total_weight).clip(0, 1)
        else:
            df["security_score"] = 0.5  # No indicators available

        return df

    def _add_security_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add individual security indicator flags."""
        # Root/jailbreak risk
        df["is_rooted_or_jailbroken"] = 0
        for col in ["IsRooted", "IsJailbroken", "IsDeviceCompromised", "CompromisedStatus"]:
            if col in df.columns:
                val = df[col].fillna(False)
                if val.dtype == "object":
                    val = val.map(lambda x: str(x).lower() in ("true", "1", "yes"))
                df["is_rooted_or_jailbroken"] = (df["is_rooted_or_jailbroken"] | val.astype(int))

        # Developer attack surface
        df["has_developer_risk"] = 0
        for col in ["IsDeveloperModeEnabled", "IsUSBDebuggingEnabled"]:
            if col in df.columns:
                val = df[col].fillna(False)
                if val.dtype == "object":
                    val = val.map(lambda x: str(x).lower() in ("true", "1", "yes"))
                df["has_developer_risk"] = (df["has_developer_risk"] | val.astype(int))

        # Encryption status
        df["is_encrypted"] = 0
        for col in ["IsEncrypted", "FileVaultEnabled"]:
            if col in df.columns:
                val = df[col].fillna(False)
                if val.dtype == "object":
                    val = val.map(lambda x: str(x).lower() in ("true", "1", "yes"))
                df["is_encrypted"] = (df["is_encrypted"] | val.astype(int))

        # Passcode protection
        df["has_passcode"] = 0
        if "HasPasscode" in df.columns:
            val = df["HasPasscode"].fillna(False)
            if val.dtype == "object":
                val = val.map(lambda x: str(x).lower() in ("true", "1", "yes"))
            df["has_passcode"] = val.astype(int)

        # Attestation passed
        df["attestation_passed"] = 0
        for col in ["IsAndroidSafetynetAttestationPassed", "KnoxAttestationStatus"]:
            if col in df.columns:
                val = df[col].fillna(False)
                if val.dtype == "object":
                    val = val.map(lambda x: str(x).lower() in ("true", "1", "yes", "passed"))
                df["attestation_passed"] = (df["attestation_passed"] | val.astype(int))

        return df

    def _add_patch_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add security patch age features."""
        # Try to find security patch date column
        patch_col = None
        for col in ["SecurityPatchLevel", "SecurityPatchDate", "LastSecurityPatch"]:
            if col in df.columns:
                patch_col = col
                break

        if patch_col is None:
            df["patch_age_days"] = np.nan
            df["patch_age_warning"] = 0
            df["patch_age_critical"] = 0
            return df

        # Parse patch date and calculate age
        reference_time = datetime.now(timezone.utc)

        def parse_patch_date(val) -> Optional[datetime]:
            if pd.isna(val):
                return None
            try:
                # Try common formats
                if isinstance(val, (datetime, pd.Timestamp)):
                    return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val
                val_str = str(val)
                # Format: "2024-01-15" or "January 2024" or "2024-01"
                for fmt in ["%Y-%m-%d", "%Y-%m", "%B %Y", "%Y%m%d"]:
                    try:
                        dt = datetime.strptime(val_str[:10], fmt)
                        return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                return None
            except Exception:
                return None

        df["_patch_date"] = df[patch_col].apply(parse_patch_date)
        df["patch_age_days"] = (reference_time - df["_patch_date"]).dt.days
        df["patch_age_days"] = df["patch_age_days"].clip(lower=0, upper=365 * 3)  # Cap at 3 years

        # Warning and critical flags
        df["patch_age_warning"] = (df["patch_age_days"] >= self.patch_age_warning_days).astype(int)
        df["patch_age_critical"] = (df["patch_age_days"] >= self.patch_age_critical_days).astype(int)

        # Clean up
        df = df.drop(columns=["_patch_date"], errors="ignore")

        return df

    def _add_risk_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk categorization based on security score and indicators."""
        # Risk scoring
        risk_score = pd.Series(0.0, index=df.index)

        # Root/jailbreak is high risk
        if "is_rooted_or_jailbroken" in df.columns:
            risk_score += df["is_rooted_or_jailbroken"] * 3

        # Developer mode is medium risk
        if "has_developer_risk" in df.columns:
            risk_score += df["has_developer_risk"] * 2

        # No encryption is medium-high risk
        if "is_encrypted" in df.columns:
            risk_score += (1 - df["is_encrypted"]) * 2

        # No passcode is medium risk
        if "has_passcode" in df.columns:
            risk_score += (1 - df["has_passcode"]) * 1.5

        # Old patches are increasing risk
        if "patch_age_critical" in df.columns:
            risk_score += df["patch_age_critical"] * 2
        if "patch_age_warning" in df.columns:
            risk_score += df["patch_age_warning"] * 1

        # Inverse of security score contributes to risk
        if "security_score" in df.columns:
            risk_score += (1 - df["security_score"]) * 2

        df["risk_score"] = risk_score

        # Categorize
        # 0-2: low, 2-5: medium, 5-8: high, 8+: critical
        df["risk_category"] = pd.cut(
            risk_score,
            bins=[-np.inf, 2, 5, 8, np.inf],
            labels=["low", "medium", "high", "critical"],
        )

        # Numeric encoding for ML
        df["risk_level"] = risk_score.clip(0, 10)

        return df

    def _add_compliance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add compliance-related features."""
        # Policy compliance
        df["is_compliant"] = 0
        for col in ["ComplianceState", "IsCompliant", "PolicyCompliant"]:
            if col in df.columns:
                val = df[col].fillna("")
                if val.dtype == "object":
                    val = val.str.lower().isin(["compliant", "true", "1", "yes"])
                else:
                    val = val.astype(bool)
                df["is_compliant"] = (df["is_compliant"] | val.astype(int))

        # Managed state
        df["is_managed"] = 0
        for col in ["IsSupervised", "IsManagedDevice", "EnrollmentStatus"]:
            if col in df.columns:
                val = df[col].fillna("")
                if val.dtype == "object":
                    val = val.str.lower().isin(["true", "1", "yes", "enrolled", "supervised"])
                else:
                    val = val.astype(bool)
                df["is_managed"] = (df["is_managed"] | val.astype(int))

        return df


def build_security_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to build security posture features.

    Args:
        df: DataFrame with device security data

    Returns:
        DataFrame with security features added
    """
    builder = SecurityFeatureBuilder()
    return builder.transform(df)


def get_security_feature_names() -> List[str]:
    """Get list of security feature names that this module generates."""
    return [
        # Composite score
        "security_score",
        # Individual flags
        "is_rooted_or_jailbroken",
        "has_developer_risk",
        "is_encrypted",
        "has_passcode",
        "attestation_passed",
        # Patch age
        "patch_age_days",
        "patch_age_warning",
        "patch_age_critical",
        # Risk
        "risk_score",
        "risk_category",
        "risk_level",
        # Compliance
        "is_compliant",
        "is_managed",
    ]


def compute_fleet_security_summary(df: pd.DataFrame) -> Dict:
    """
    Compute fleet-wide security summary statistics.

    Args:
        df: DataFrame with security features

    Returns:
        Dictionary with fleet security metrics
    """
    # Build features if not present
    if "security_score" not in df.columns:
        df = build_security_features(df)

    summary = {
        "total_devices": len(df),
    }

    # Average security score
    if "security_score" in df.columns:
        summary["avg_security_score"] = float(df["security_score"].mean())
        summary["security_score_std"] = float(df["security_score"].std())
        summary["low_security_count"] = int((df["security_score"] < 0.5).sum())

    # Risk distribution
    if "risk_category" in df.columns:
        risk_dist = df["risk_category"].value_counts().to_dict()
        summary["risk_distribution"] = {str(k): int(v) for k, v in risk_dist.items()}

    # Specific vulnerabilities
    if "is_rooted_or_jailbroken" in df.columns:
        summary["rooted_jailbroken_count"] = int(df["is_rooted_or_jailbroken"].sum())
        summary["rooted_jailbroken_pct"] = float(df["is_rooted_or_jailbroken"].mean() * 100)

    if "is_encrypted" in df.columns:
        summary["unencrypted_count"] = int((1 - df["is_encrypted"]).sum())
        summary["encryption_rate_pct"] = float(df["is_encrypted"].mean() * 100)

    if "patch_age_critical" in df.columns:
        summary["critical_patch_age_count"] = int(df["patch_age_critical"].sum())

    if "is_compliant" in df.columns:
        summary["compliance_rate_pct"] = float(df["is_compliant"].mean() * 100)

    return summary
