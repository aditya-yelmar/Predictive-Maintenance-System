def maintenance_recommendation(
    rul,
    anomaly
):

    if anomaly:
        return {
            "priority": "CRITICAL",
            "action": "Immediate Maintenance"
        }

    if rul < 20:
        return {
            "priority": "HIGH",
            "action": "Within 24 Hours"
        }

    if rul < 50:
        return {
            "priority": "MEDIUM",
            "action": "Within 7 Days"
        }

    return {
        "priority": "LOW",
        "action": "Routine Monitoring"
    }


if __name__ == "__main__":

    print(
        maintenance_recommendation(
            rul=15,
            anomaly=True
        )
    )