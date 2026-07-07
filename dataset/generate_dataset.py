"""Generate a synthetic dataset of 120 customer support emails.

Each example contains structured metadata and a gold-standard reply.
Covers 14 support categories with realistic scenarios.
"""

import json
import random
import uuid
from pathlib import Path


INTENTS = [
    "refund", "billing", "login_issue", "subscription", "cancellation",
    "complaint", "bug_report", "feature_request", "integration",
    "enterprise_sales", "password_reset", "pricing", "account_verification",
    "positive_feedback",
]

URGENCY_LEVELS = ["low", "medium", "high"]

TONES = ["frustrated", "neutral", "polite", "urgent", "appreciative", "confused"]


TEMPLATES: dict[str, list[dict]] = {
    "refund": [
        {
            "body": "I recently purchased {product} on {date} but it doesn't work as advertised. I'd like a full refund. The order number is {order_id}.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["process refund", "confirm refund timeline"],
            "gold": "I'm sorry {product} didn't meet your expectations. I've initiated a full refund for order {order_id}. You should see the funds back within 5-7 business days. Let me know if you need anything else.",
        },
        {
            "body": "Hi, I ordered {product} but received the wrong version. Can you help me get a refund or exchange? Order #{order_id}.",
            "tone": "polite",
            "urgency": "medium",
            "actions": ["process refund or exchange", "confirm correct version"],
            "gold": "I apologize for the mix-up with your order. I've started a refund for order {order_id}. Once the return is received, we'll ship the correct version at no extra cost. Does that work?",
        },
        {
            "body": "This is the third time I'm requesting a refund for {product}. Order {order_id}. I've been ignored twice. I want my money back NOW.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["apologize for delay", "process refund immediately", "escalate if needed"],
            "gold": "I sincerely apologize for the delays. I'm personally handling this now — your refund for {product} (order {order_id}) has been processed and should reflect within 3-5 business days. I've also added a $20 credit for the inconvenience.",
        },
        {
            "body": "Can I get a refund for {product}? I bought it {days} days ago. Order: {order_id}.",
            "tone": "neutral",
            "urgency": "low",
            "actions": ["check refund policy", "process refund"],
            "gold": "Thanks for reaching out. Since your purchase is within our {days}-day refund window, I've processed a full refund for order {order_id}. You'll see the amount in 5-7 business days.",
        },
    ],
    "billing": [
        {
            "body": "I was charged twice for my {plan} subscription this month. My account email is {email}. Please fix this.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["investigate duplicate charge", "issue refund for extra charge"],
            "gold": "I see the duplicate charge on your account. I've refunded the extra {plan} payment — it should appear in 3-5 business days. Apologies for the billing error.",
        },
        {
            "body": "My invoice for {month} doesn't match what I was quoted. The amount is ${amount} but I was told ${expected}. Can you clarify?",
            "tone": "confused",
            "urgency": "medium",
            "actions": ["review invoice", "explain discrepancy", "adjust if incorrect"],
            "gold": "Let me look into this. The ${amount} charge includes a prorated upgrade fee from mid-{month}. I've emailed you an itemized breakdown. If the upgrade wasn't authorized, I'll reverse it and adjust the invoice.",
        },
        {
            "body": "My {plan} plan is showing as past due but I paid on {date}. Please check. Invoice #{invoice_id}.",
            "tone": "polite",
            "urgency": "medium",
            "actions": ["check payment status", "update billing system"],
            "gold": "I checked our system — your payment on {date} was received but not applied due to a processing delay. I've corrected this and your {plan} plan is now active. Sorry for the confusion.",
        },
    ],
    "login_issue": [
        {
            "body": "I can't log into my account with email {email}. It says 'invalid credentials' but my password is correct. I've already tried resetting it twice.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["check account status", "reset password", "verify email"],
            "gold": "I've unlocked your account and sent a password reset link to {email}. After resetting, please clear your browser cache. If the issue persists, let me know and I'll escalate to our engineering team.",
        },
        {
            "body": "Getting a 2FA error when trying to log in. The code I receive doesn't work. Account: {email}.",
            "tone": "neutral",
            "urgency": "medium",
            "actions": ["troubleshoot 2FA", "provide backup codes"],
            "gold": "Let's get this sorted. Try these steps: 1) Ensure your device clock is synced, 2) Use one of the backup codes I've attached. I've also disabled and re-enabled 2FA on your account — please try again.",
        },
        {
            "body": "Hi team, I created an account with {email} but never received the verification email. I can't log in.",
            "tone": "polite",
            "urgency": "low",
            "actions": ["resend verification email", "confirm account status"],
            "gold": "I've manually verified your account and resent the confirmation email to {email}. You should be able to log in now. Check your spam folder if you don't see it.",
        },
    ],
    "subscription": [
        {
            "body": "I'd like to upgrade from {current_plan} to {new_plan}. Can you help me do that without losing my data?",
            "tone": "neutral",
            "urgency": "low",
            "actions": ["process upgrade", "ensure data preservation"],
            "gold": "Great choice! I've upgraded your account from {current_plan} to {new_plan}. All your data is intact. The prorated difference of ${amount} will be charged today. Welcome to {new_plan}!",
        },
        {
            "body": "My {plan} subscription auto-renewed but I didn't want to continue. I need to cancel and get a refund for the renewed amount.",
            "tone": "frustrated",
            "urgency": "medium",
            "actions": ["cancel subscription", "refund renewal"],
            "gold": "I understand — I've cancelled your subscription and processed a refund for the renewal charge. Your access will continue until the end of the current billing period. You'll receive a confirmation email shortly.",
        },
        {
            "body": "Does the {plan} plan include access to all features? I'm confused about what's included in each tier. Can someone explain?",
            "tone": "confused",
            "urgency": "low",
            "actions": ["explain plan features", "compare tiers"],
            "gold": "Happy to clarify! The {plan} plan includes all core features. Here's what's included: [feature list]. If you need advanced reporting and SSO, you'd want the Pro tier. Would you like a detailed comparison sheet?",
        },
    ],
    "cancellation": [
        {
            "body": "I want to cancel my account immediately. I've been having constant issues with {feature} and I'm done.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["cancel account", "offer retention", "export data"],
            "gold": "I'm sorry to hear about the issues with {feature}. I'd like to help resolve them before we proceed with cancellation. If you'd still like to cancel, I've initiated the process and a data export link will be emailed to you within 24 hours.",
        },
        {
            "body": "Please cancel my {plan} subscription at the end of this billing cycle. No issues, just don't need it anymore.",
            "tone": "polite",
            "urgency": "low",
            "actions": ["schedule cancellation", "confirm end date"],
            "gold": "I've scheduled your {plan} cancellation for the end of this billing cycle ({date}). Your data will be available for 30 days after that if you change your mind. Is there anything else I can help with?",
        },
    ],
    "complaint": [
        {
            "body": "Your customer support is terrible. I've been waiting {days} days for a response to my previous email about {issue}. This is unacceptable for a paid service.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["apologize for delay", "resolve original issue", "compensate"],
            "gold": "I sincerely apologize for the {days}-day wait — that's not the experience we want to provide. I've prioritized your case and personally looked into the {issue}. It's now resolved, and I've added a month of credit to your account. Please give me a chance to make this right.",
        },
        {
            "body": "The {feature} feature stopped working after the latest update. It's been {days} days and no fix. This affects my entire workflow.",
            "tone": "urgent",
            "urgency": "high",
            "actions": ["investigate bug", "provide workaround", "set fix timeline"],
            "gold": "I've confirmed the regression in {feature} after our latest release. Our engineering team has identified the root cause and is deploying a fix within 24 hours. In the meantime, here's a workaround: [steps]. I'll notify you as soon as it's resolved.",
        },
    ],
    "bug_report": [
        {
            "body": "When I click 'export' on the {page} page, the CSV file is empty. This started happening yesterday. Using Chrome on Windows.",
            "tone": "neutral",
            "urgency": "medium",
            "actions": ["log bug report", "investigate export issue", "provide workaround"],
            "gold": "Thanks for the detailed report. I've reproduced the issue — the export endpoint is returning an empty payload. Engineering is deploying a fix. For now, try using the API export endpoint: [endpoint]. I'll update you when the UI is fixed.",
        },
        {
            "body": "The mobile app crashes whenever I try to upload a photo. I'm on iOS {version}, latest app version.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["log bug", "investigate crash", "offer alternative"],
            "gold": "I've logged this with our mobile team. The crash is related to the image processing library on iOS {version}. A patch is in review. Meanwhile, you can upload photos via the web interface. I'll email you when the fix is live.",
        },
        {
            "body": "Notifications aren't being delivered. I checked my settings and everything is enabled. Missing notifications for {event_type}.",
            "tone": "neutral",
            "urgency": "medium",
            "actions": ["investigate notification system", "check delivery logs"],
            "gold": "I found the issue — there was a delay in our notification queue. All pending notifications have been flushed. You should see them now. I've also added monitoring to catch this earlier in the future.",
        },
    ],
    "feature_request": [
        {
            "body": "It would be great if you added dark mode to the dashboard. I work late nights and the white background is harsh.",
            "tone": "polite",
            "urgency": "low",
            "actions": ["log feature request", "share with product team"],
            "gold": "Great suggestion! Dark mode is actually on our Q3 roadmap. I've added your vote to the feature request. I'll share your feedback with the product team. Any other specific areas you'd like to see themed?",
        },
        {
            "body": "We need bulk user import via CSV for the enterprise plan. Manually adding {count} users is not feasible.",
            "tone": "urgent",
            "urgency": "high",
            "actions": ["log feature request", "provide current workaround"],
            "gold": "I understand the pain point. Bulk CSV import is being developed for release next quarter. In the meantime, I can help you set up our API-based user provisioning — I'll send you the docs and a sample script. Would that work?",
        },
        {
            "body": "Can you add integration with {tool_name}? Our entire team uses it and manual data transfer is slowing us down.",
            "tone": "polite",
            "urgency": "medium",
            "actions": ["log integration request", "check existing integrations"],
            "gold": "Good news — we actually have a {tool_name} integration in beta. I can enable it for your workspace right now. If you run into any issues, our integration specialist can schedule a setup call. Want me to proceed?",
        },
    ],
    "integration": [
        {
            "body": "I set up the Slack integration but messages aren't coming through. I followed the setup guide but something is wrong.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["diagnose integration", "verify configuration"],
            "gold": "Let me check your Slack integration. The most common issue is the webhook URL expiring. I've re-authorized the connection and test messages should appear in #general now. Let me know if you see them.",
        },
        {
            "body": "Does your API support webhooks for new ticket events? We want to sync with our internal system. Documentation wasn't clear.",
            "tone": "neutral",
            "urgency": "medium",
            "actions": ["explain webhook support", "share documentation"],
            "gold": "Yes, we support webhooks for all ticket events including creation, updates, and resolution. I've enabled webhook access for your account. Here's the endpoint URL and event format: [docs]. Let me know if you need a sample payload.",
        },
    ],
    "enterprise_sales": [
        {
            "body": "We're a team of {count} and interested in your enterprise plan. Need SSO, audit logs, and dedicated support. Can you share pricing?",
            "tone": "neutral",
            "urgency": "medium",
            "actions": ["share enterprise pricing", "schedule demo", "discuss requirements"],
            "gold": "Thanks for your interest! Our enterprise plan includes SSO (SAML/OIDC), audit logs, SLA guarantees, and a dedicated success manager. I'd love to schedule a 30-minute call to walk through your requirements and provide a tailored quote. When works for you?",
        },
        {
            "body": "Hi, we're evaluating your platform for our {industry} company. Need to understand your SOC 2 compliance and data residency options before we proceed.",
            "tone": "polite",
            "urgency": "medium",
            "actions": ["share compliance docs", "discuss data residency"],
            "gold": "I can help with that. We're SOC 2 Type II certified (report available under NDA) and offer data residency in US, EU, and APAC regions. I'll send you our security whitepaper and can set up an intro call with our security team. Does that work?",
        },
    ],
    "password_reset": [
        {
            "body": "I forgot my password and the reset link isn't arriving. I've checked spam. Email: {email}.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["trigger password reset", "verify email delivery"],
            "gold": "I've manually triggered a reset link to {email}. It should arrive within 2 minutes. If it doesn't, there may be a typo in the email on your account — I can help you verify. Let me know either way.",
        },
        {
            "body": "Reset link expired before I could use it. Can you send another one? Email: {email}.",
            "tone": "neutral",
            "urgency": "medium",
            "actions": ["send new reset link"],
            "gold": "I've sent a new reset link with a 2-hour expiry. Click it within that window and you'll be good. Let me know if you need any further help.",
        },
    ],
    "pricing": [
        {
            "body": "Is there a student discount? I'm a university student and the {plan} plan is a bit expensive for me.",
            "tone": "polite",
            "urgency": "low",
            "actions": ["share discount options", "suggest suitable plan"],
            "gold": "Great question! We offer a 50% student discount for verified .edu email addresses. Sign up with your university email and apply code STUDENT50 at checkout. That brings the {plan} plan to ${discounted_price}/month.",
        },
        {
            "body": "Your pricing page says {price1}/month but after signing up I see {price2}. What am I missing?",
            "tone": "confused",
            "urgency": "medium",
            "actions": ["clarify pricing", "check applied plan"],
            "gold": "I see the confusion - the {price1} is our annual billing price (billed yearly). The {price2} is the monthly billing option. If you switch to annual you can save around 20%. I can switch that for you if you would like.",
        },
    ],
    "account_verification": [
        {
            "body": "My account was suspended and I need it restored. I think it's because of a payment issue. Email: {email}.",
            "tone": "frustrated",
            "urgency": "high",
            "actions": ["check suspension reason", "restore account"],
            "gold": "I've checked your account — it was flagged by our billing system after a failed payment. I've reactivated it and your data is all intact. I can help update your payment method to prevent future issues.",
        },
        {
            "body": "I need to verify my business account. I uploaded the documents {days} days ago but haven't heard back. Account: {email}.",
            "tone": "neutral",
            "urgency": "medium",
            "actions": ["check verification status", "approve or request info"],
            "gold": "Apologies for the delay. I've reviewed your documents — everything looks good. Your business account is now verified. You now have access to all enterprise features including team management and audit logs.",
        },
    ],
    "positive_feedback": [
        {
            "body": "Just wanted to say your support team is amazing! {agent_name} helped me resolve my {issue} in minutes. This is why I stay with your product.",
            "tone": "appreciative",
            "urgency": "low",
            "actions": ["thank customer", "share feedback with team"],
            "gold": "Wow, thank you for the kind words! I've shared your feedback with {agent_name} and the team — it made our day. We're glad to have you as a customer. If there's ever anything we can do, you know where to find us.",
        },
        {
            "body": "The new {feature} feature is exactly what we needed. Our team productivity has improved significantly. Keep up the great work!",
            "tone": "appreciative",
            "urgency": "low",
            "actions": ["thank customer", "share feedback with product team"],
            "gold": "This means a lot to the team! I've shared your feedback with our product team — hearing how {feature} is making a real impact is why we do what we do. Any other features on your wishlist?",
        },
    ],
}


def _pick(seq: list) -> str:
    return random.choice(seq)


def _fill(template: str) -> str:
    subs = {
        "product": _pick([
            "ProjectPro", "DataSync", "CloudBase", "TaskFlow", "Analytix",
            "TeamHub", "DocuSign Pro", "MailChimp Pro", "Slack Premium",
            "Notion Team", "Linear", "Figma Enterprise",
        ]),
        "date": _pick([f"last {d}" for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]] + ["January 15th", "February 3rd", "March 20th", "April 10th", "May 5th", "June 1st", "yesterday"]),
        "order_id": "ORD-" + str(random.randint(100000, 999999)),
        "days": str(random.randint(3, 30)),
        "email": _pick(["alex.johnson", "sarah.chen", "mike.rivera", "emma.thompson", "james.wilson", "priya.patel", "carlos.garcia"]) + "@" + _pick(["gmail.com", "outlook.com", "company.com", "proton.me"]),
        "plan": _pick(["Starter", "Basic", "Pro", "Business", "Enterprise"]),
        "month": _pick(["January", "February", "March", "April", "May", "June", "July", "August", "September"]),
        "amount": str(random.randint(49, 999)),
        "expected": str(random.randint(29, 79)),
        "current_plan": _pick(["Free", "Starter", "Basic"]),
        "new_plan": _pick(["Pro", "Business", "Enterprise"]),
        "feature": _pick(["reporting", "export", "dashboard", "search", "notification", "API", "automation", "analytics"]),
        "page": _pick(["dashboard", "reports", "settings", "analytics", "team", "integrations", "billing"]),
        "issue": _pick(["a broken feature", "a missing integration", "an incorrect charge", "an account lockout", "a data export problem", "a performance issue", "a security concern", "a setup error"]),
        "invoice_id": "INV-" + str(random.randint(10000, 99999)),
        "version": _pick(["17.4", "18.0", "18.1"]),
        "event_type": _pick(["task assignments", "mentions", "deadline reminders", "status changes"]),
        "count": str(random.randint(150, 2000)),
        "tool_name": _pick(["Jira", "Asana", "Salesforce", "HubSpot", "Zendesk", "GitHub", "GitLab", "Trello"]),
        "industry": _pick(["fintech", "healthcare", "e-commerce", "SaaS", "logistics"]),
        "price1": _pick(["$29", "$49", "$79", "$99"]),
        "price2": _pick(["$35", "$59", "$95", "$119"]),
        "discounted_price": str(random.randint(14, 49)),
        "agent_name": _pick(["Sarah", "Mike", "Priya", "James", "Emma", "Carlos", "Alex"]),
    }
    return template.format(**subs)


def generate_dataset(count: int = 120, seed: int = 42) -> list[dict]:
    random.seed(seed)
    examples = []

    for i in range(count):
        intent = random.choices(
            INTENTS,
            weights=[10, 10, 8, 8, 6, 8, 8, 8, 6, 4, 6, 6, 6, 4],
            k=1
        )[0]

        template = _pick(TEMPLATES[intent])
        body = _fill(template["body"])
        gold = _fill(template["gold"])
        tone = template["tone"]

        urgency = template["urgency"]
        if intent == "bug_report" and "crash" in body.lower():
            urgency = "high"
        if intent == "feature_request":
            urgency = "low"

        example = {
            "id": f"email-{i+1:03d}",
            "customer_email": body,
            "intent": intent,
            "urgency": urgency,
            "tone": tone,
            "expected_actions": template["actions"],
            "gold_reply": gold,
        }
        examples.append(example)

    return examples


def save_dataset(examples: list[dict], path: str = "dataset/emails.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} examples to {path}")


if __name__ == "__main__":
    dataset = generate_dataset(120)
    save_dataset(dataset)
