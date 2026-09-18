# CloudSync Pro — Subscription & Support Policy (sample)

## 1. Subscription tiers

CloudSync Pro is offered in three tiers: Starter, Team, and Enterprise.

The Starter tier includes 50GB of storage and supports up to 3 connected
devices. It does not include priority support or version history beyond 7
days.

The Team tier includes 500GB of storage, up to 25 connected devices, and 30
days of version history. Priority support is included, with a guaranteed
first response within 4 business hours.

The Enterprise tier includes unlimited storage, unlimited devices, and 1 year
of version history. Enterprise customers get a dedicated account manager and
a guaranteed first response within 1 business hour.

Note: the old Business tier (discontinued in 2023) offered 500GB of storage
with only 90 days of version history — do not confuse it with the current
Team tier, which has different limits.

## 2. Cancellation and refunds

Subscriptions can be cancelled at any time from account settings. Cancelling
does not trigger an automatic refund. Customers on a monthly plan are not
eligible for prorated refunds under any circumstance.

Customers on an annual plan may request a refund within 14 days of the annual
renewal date. Refund requests after 14 days are only granted if the account
had less than 1GB of stored data at the time of cancellation.

Enterprise contracts are governed by the separate Master Services Agreement
and are not eligible for the 14-day refund window described above.

## 3. Data retention after cancellation

After cancellation, account data is retained for 30 days in a recoverable
state. After 30 days, data enters a 60-day soft-delete window during which it
can only be restored by contacting support directly — it is no longer
recoverable via the app itself. After a total of 90 days from cancellation,
all data is permanently deleted and cannot be recovered under any
circumstances.

## 4. Support escalation

Standard support tickets (Starter tier) are handled on a best-effort basis
with no guaranteed response time.

Priority tickets (Team and Enterprise) are automatically escalated to a
senior engineer if unresolved after 24 hours.

Security-related tickets — account compromise, suspicious login activity, or
suspected data exposure — are escalated immediately regardless of tier, and
are handled by the security response team rather than general support.

## 5. API rate limits

The REST API allows 100 requests per minute on the Team tier and 1,000
requests per minute on the Enterprise tier. The Starter tier does not have
API access.

Exceeding the rate limit returns error code RL-429 and a 60 second cooldown.
This is distinct from error AUTH-401, which indicates an invalid or expired
API token rather than a rate limit issue — these two are commonly confused in
support tickets.