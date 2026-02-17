"""Generate 500 synthetic historical fraud case studies for RAG demonstration."""

import json
import random
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)

# Location pools
us_cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"]
intl_cities = ["Tokyo", "London", "Paris", "Dubai", "Singapore", "Sydney", "Mumbai", "Toronto", "Hong Kong", "Berlin"]
all_cities = us_cities + intl_cities

# Categories
categories = ["grocery", "restaurant", "gas", "pharmacy", "electronics", "clothing", "entertainment", "travel", "hotel"]
high_risk_categories = ["gift_cards", "crypto", "wire_transfer", "prepaid_cards", "money_order"]

# Devices
devices = ["mobile", "desktop", "tablet"]

def generate_velocity_case(case_id: int) -> dict:
    """Generate velocity attack fraud case."""
    user_id = f"U_{random.randint(1000, 9999)}"
    base_amount = round(random.uniform(20, 150), 2)
    location = random.choice(us_cities)
    device = random.choice(devices)
    txn_count = random.randint(9, 25)

    return {
        "case_id": f"CASE_{case_id:04d}",
        "fraud_type": "velocity",
        "summary": f"{txn_count} transactions in 4 minutes - card testing attack",
        "transaction_pattern": f"User {user_id}: {txn_count} transactions averaging ${base_amount} within 4 minutes",
        "location": location,
        "device": device,
        "indicators": [
            f"Extremely high velocity: {txn_count} transactions in 4 minutes",
            f"All transactions same amount range (${base_amount-10:.2f}-${base_amount+10:.2f})",
            "Rapid succession suggests automated bot",
            "Testing stolen card before larger fraud"
        ],
        "reasoning": f"Normal users make 1-2 transactions per hour. {txn_count} transactions in 4 minutes indicates automated card testing. Pattern matches known bot behavior where fraudsters test card validity before making large purchases.",
        "outcome": "Confirmed fraud - card was stolen, account blocked, $0 loss",
        "amount_range": f"${base_amount-10:.2f}-${base_amount+10:.2f}",
        "time_window": "4 minutes",
        "severity": "critical"
    }

def generate_amount_anomaly_case(case_id: int) -> dict:
    """Generate amount anomaly fraud case."""
    user_id = f"U_{random.randint(1000, 9999)}"
    normal_amount = round(random.uniform(30, 100), 2)
    fraud_amount = round(random.uniform(2000, 8000), 2)
    location = random.choice(us_cities)
    category = random.choice(["electronics", "jewelry", "luxury_goods"])

    return {
        "case_id": f"CASE_{case_id:04d}",
        "fraud_type": "amount_anomaly",
        "summary": f"${fraud_amount} purchase - 50x higher than normal ${normal_amount} average",
        "transaction_pattern": f"User {user_id}: Normal spending ${normal_amount}/txn, suddenly ${fraud_amount} at {category} store",
        "location": location,
        "device": random.choice(devices),
        "indicators": [
            f"Amount ${fraud_amount} is {int(fraud_amount/normal_amount)}x user's average",
            f"No prior large purchases in 6-month history",
            f"High-risk category: {category}",
            "No gradual spending increase pattern"
        ],
        "reasoning": f"User's 6-month transaction history shows average of ${normal_amount} for groceries and essentials. Sudden ${fraud_amount} purchase at {category} store with no prior large purchases is statistically anomalous (>5 standard deviations). Classic account takeover pattern.",
        "outcome": f"Confirmed fraud - stolen card, merchant refunded ${fraud_amount}",
        "amount_range": f"${fraud_amount}",
        "normal_spending": f"${normal_amount}",
        "severity": "high"
    }

def generate_geographic_case(case_id: int) -> dict:
    """Generate geographic outlier fraud case."""
    user_id = f"U_{random.randint(1000, 9999)}"
    home_city = random.choice(us_cities)
    fraud_city = random.choice(intl_cities)
    amount = round(random.uniform(500, 3000), 2)
    time_gap_hours = random.randint(1, 4)

    return {
        "case_id": f"CASE_{case_id:04d}",
        "fraud_type": "geographic",
        "summary": f"{home_city} to {fraud_city} in {time_gap_hours} hours - impossible travel",
        "transaction_pattern": f"User {user_id}: Purchase in {home_city}, then ${amount} in {fraud_city} {time_gap_hours}h later",
        "location": f"{home_city} → {fraud_city}",
        "device": random.choice(devices),
        "indicators": [
            f"Transaction in {home_city} at 10:00 AM",
            f"Transaction in {fraud_city} at {10+time_gap_hours}:00 PM same day",
            f"Distance: ~{random.randint(5000, 9000)} miles",
            f"Time gap: {time_gap_hours} hours (impossible even by plane)",
            "No flight or hotel bookings in between"
        ],
        "reasoning": f"Physical travel from {home_city} to {fraud_city} requires minimum {random.randint(12, 18)} hours including flight time. User made purchase in {home_city} then {fraud_city} only {time_gap_hours} hours later - physically impossible. No travel-related transactions (flights, hotels) detected. Clear indicator of card cloning or stolen credentials.",
        "outcome": "Confirmed fraud - card cloned and used internationally",
        "amount_range": f"${amount}",
        "time_window": f"{time_gap_hours} hours",
        "severity": "critical"
    }

def generate_account_takeover_case(case_id: int) -> dict:
    """Generate account takeover fraud case."""
    user_id = f"U_{random.randint(1000, 9999)}"
    normal_device = random.choice(["iPhone 14", "Samsung Galaxy S23"])
    fraud_device = random.choice(["Android Emulator", "Linux Desktop", "Unknown Device"])
    location = random.choice(us_cities)
    fraud_location = random.choice(intl_cities)
    amount = round(random.uniform(1000, 5000), 2)

    return {
        "case_id": f"CASE_{case_id:04d}",
        "fraud_type": "account_takeover",
        "summary": f"Device change + behavioral shift + ${amount} in gift cards",
        "transaction_pattern": f"User {user_id}: {normal_device} → {fraud_device}, groceries → gift cards",
        "location": f"{location} → {fraud_location}",
        "device": f"{normal_device} changed to {fraud_device}",
        "indicators": [
            f"Device change: {normal_device} → {fraud_device}",
            f"Location change: {location} → {fraud_location}",
            "Password reset 15 minutes before transaction",
            "Email address changed",
            f"Category shift: groceries/essentials → gift cards",
            f"Spending spike: ${amount} (3x normal)"
        ],
        "reasoning": f"User typically purchases groceries from {location} on {normal_device}. Account shows password reset followed by email change, then immediate ${amount} gift card purchase from {fraud_location} on {fraud_device}. Classic account takeover sequence: compromise → secure access → extract value via gift cards. Device fingerprint mismatch confirms unauthorized access.",
        "outcome": f"Confirmed fraud - credentials phished, account recovered, ${amount} charged back",
        "amount_range": f"${amount}",
        "category_shift": "groceries → gift_cards",
        "severity": "critical"
    }

def generate_mixed_indicator_case(case_id: int) -> dict:
    """Generate fraud case with multiple indicators."""
    user_id = f"U_{random.randint(1000, 9999)}"
    amount = round(random.uniform(800, 4000), 2)
    location = random.choice(intl_cities)
    hour = random.choice([2, 3, 4])  # Middle of night

    fraud_types = ["velocity", "amount_anomaly", "geographic", "account_takeover"]
    primary_type = random.choice(fraud_types)

    return {
        "case_id": f"CASE_{case_id:04d}",
        "fraud_type": primary_type,
        "summary": f"Multiple red flags: {location} at {hour}AM, ${amount}, new device, VPN",
        "transaction_pattern": f"User {user_id}: ${amount} at {hour}AM from {location} via VPN on new device",
        "location": location,
        "device": "New Android device (never seen before)",
        "indicators": [
            f"Unusual hour: {hour}:00 AM (user normally asleep)",
            f"New location: {location} (user normally in US)",
            f"Amount: ${amount} (2.5x normal spending)",
            "VPN detected (hiding true location)",
            "New device fingerprint",
            "High-risk merchant: electronics + gift cards combo",
            "Rush delivery requested"
        ],
        "reasoning": f"Multiple fraud indicators compound the risk. Transaction at {hour}AM from {location} when user is US-based and typically asleep. VPN usage hides true origin. New device fingerprint with no prior auth. ${amount} purchase combining electronics and gift cards (common fraud pattern). Rush delivery suggests urgency before card is blocked. Individually suspicious, collectively conclusive.",
        "outcome": f"Confirmed fraud - stolen credentials + VPN, blocked before shipment",
        "amount_range": f"${amount}",
        "time_window": f"{hour}:00 AM",
        "severity": "critical"
    }

def generate_card_testing_case(case_id: int) -> dict:
    """Generate card testing fraud case."""
    user_id = f"U_{random.randint(1000, 9999)}"
    test_amounts = ["$0.01", "$1.00", "$0.50"]
    location = random.choice(us_cities)

    return {
        "case_id": f"CASE_{case_id:04d}",
        "fraud_type": "velocity",
        "summary": "Card testing: 15 micro-transactions then $2500 purchase",
        "transaction_pattern": f"User {user_id}: 15x {random.choice(test_amounts)} auth checks, then $2500",
        "location": location,
        "device": "Multiple devices",
        "indicators": [
            f"15 authorization attempts at {random.choice(test_amounts)} each",
            "Different merchants, same time window",
            "Most auths: declined, 3 approved",
            "Followed by $2500 purchase after successful test",
            "Time gap: 2 minutes between testing and large purchase"
        ],
        "reasoning": "Fraudster testing stolen card validity with small authorization requests. Pattern shows automated testing across multiple merchants to find one that approves. Once card validates, immediate large purchase attempted. Classic card testing attack - if small amounts authorize, card is active and can be exploited.",
        "outcome": "Confirmed fraud - card testing detected, large purchase blocked",
        "amount_range": "$0.01-$2500",
        "time_window": "15 minutes",
        "severity": "high"
    }

def generate_shipping_fraud_case(case_id: int) -> dict:
    """Generate shipping/address fraud case."""
    user_id = f"U_{random.randint(1000, 9999)}"
    billing_city = random.choice(us_cities)
    shipping_city = random.choice(us_cities + intl_cities)
    amount = round(random.uniform(1200, 5000), 2)

    return {
        "case_id": f"CASE_{case_id:04d}",
        "fraud_type": "account_takeover",
        "summary": f"${amount} shipped to new address, billing/shipping mismatch",
        "transaction_pattern": f"User {user_id}: Billing {billing_city}, shipping {shipping_city}, overnight delivery",
        "location": billing_city,
        "device": random.choice(devices),
        "indicators": [
            f"Billing address: {billing_city}",
            f"Shipping address: {shipping_city} (never used before)",
            f"Address used by 25 other accounts (known drop address)",
            "Overnight shipping requested",
            f"High value: ${amount}",
            "First purchase to this address",
            "No gift message (not a gift)"
        ],
        "reasoning": f"Shipping address in {shipping_city} never previously used by customer. Address appears in fraud database as 'drop address' linked to 25 other compromised accounts. {billing_city} to {shipping_city} mismatch without gift indication is red flag. Overnight shipping urgency suggests fraudster wants items before card is blocked. Classic reshipping fraud pattern.",
        "outcome": f"Confirmed fraud - known drop address, shipment intercepted",
        "amount_range": f"${amount}",
        "severity": "high"
    }

# Generate 500 cases with distribution
cases = []

# Distribution: 125 cases per primary fraud type
distributions = [
    (generate_velocity_case, 90),
    (generate_amount_anomaly_case, 90),
    (generate_geographic_case, 90),
    (generate_account_takeover_case, 90),
    (generate_mixed_indicator_case, 80),
    (generate_card_testing_case, 30),
    (generate_shipping_fraud_case, 30)
]

case_id = 1
for generator_func, count in distributions:
    for _ in range(count):
        cases.append(generator_func(case_id))
        case_id += 1

# Save to JSON
output_file = "data/historical_fraud_cases.json"
with open(output_file, 'w') as f:
    json.dump(cases, f, indent=2)

print(f"✅ Generated {len(cases)} historical fraud cases")
print(f"📁 Saved to: {output_file}")

# Print distribution
from collections import Counter
fraud_types = Counter([case['fraud_type'] for case in cases])
print("\n📊 Distribution by fraud type:")
for fraud_type, count in sorted(fraud_types.items()):
    print(f"   {fraud_type}: {count} cases")

# Calculate estimated token count
total_chars = sum([len(json.dumps(case)) for case in cases])
estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars
print(f"\n💰 Estimated tokens for all cases: ~{estimated_tokens:,}")
print(f"   (This will make naive approach expensive!)")
