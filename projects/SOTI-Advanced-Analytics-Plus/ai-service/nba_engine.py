import logging

logger = logging.getLogger("SAAP.NBA")


class NextBestActionEngine:
    """
    SAAP AI: Recommends surgical remediations based on grounded anomaly scores.
    """
    def recommend(self, anomaly):
        if not isinstance(anomaly, dict):
            logger.error("Invalid anomaly input: expected dict, got %s", type(anomaly).__name__)
            return {"action": "Manual Investigation Required", "target": "Admin"}

        anomaly_type = anomaly.get("type", "unknown")
        root_cause = anomaly.get("root_cause_hint", "")
        summary = anomaly.get("summary", "")
        cohort = anomaly.get("affected_cohort", "Unknown")

        if anomaly_type == "connectivity" and "Profile" in root_cause:
            return {
                "action": "Rollback Profile",
                "target": cohort,
                "impact": "High - Restores Site Connectivity",
                "risk": "Low",
            }

        if anomaly_type == "hardware" and "Thermal" in summary:
            return {
                "action": "Throttle Background Apps",
                "target": cohort,
                "impact": "Medium - Prevents Battery Degradation",
                "risk": "Zero",
            }

        logger.info("No specific rule for anomaly type=%s; recommending manual investigation.", anomaly_type)
        return {"action": "Manual Investigation Required", "target": "Admin"}


if __name__ == "__main__":
    nba = NextBestActionEngine()
    print("NBA Engine Initialized.")
