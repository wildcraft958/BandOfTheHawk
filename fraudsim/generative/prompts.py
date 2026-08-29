"""The prompts that render each text vertical.

Three verticals present text to a control: a disputed charge, a support request,
a refund claim. Each is built from a system turn and a user turn filled with
facts the simulator supplies — the amount, the merchant, the date, a persona.

Two disciplines make the resulting data honest.

**Fraud and benign share a template.** The negative twin of a fraudulent dispute
is a real customer disputing a charge they genuinely did not make; the letter
reads the same, and the only difference is whether the underlying transaction
was authorised — a fact that lives in the episode label, not in the text. If the
two classes came from different templates a text model would learn the template
rather than the fraud, so the benign twin is the same prompt with the
adversarial facts removed, never a different prompt.

**The tier is richness, not intent.** A higher capability tier asks for more
specific, consistent, well-structured text — a timeline, pre-empted objections,
cross-referenced detail — not for anything more adversarial. That ladder is
ordinal and measurable, which is what the capability-tier claim needs, and it
never requires the model to believe it is doing anything wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM = (
    "You are writing on behalf of a bank customer. Produce only the letter or "
    "message text — no preamble, no explanation, no subject line unless asked. "
    "Write naturally, as a real person would: plain language, occasional "
    "imperfection, specific concrete detail. Do not use placeholders like "
    "[NAME] or [DATE]; use the details provided. Length: {length_hint}."
)

# The capability ladder. Length and the extra instructions grow together, so a
# higher tier produces text that is harder to cluster and richer in checkable
# detail — exactly what the scores read.
LENGTH_HINT = {
    0: "2 to 3 terse sentences",
    1: "a short paragraph",
    2: "3 to 4 paragraphs with a clear order of events",
    3: "4 to 6 paragraphs with a clear timeline and specific references",
}

TIER_EXTRAS = {
    "dispute": {
        3: (
            "Reference the customer's long account tenure and that this has "
            "never happened before, and request a written response within 10 "
            "business days."
        ),
    },
    "ticket": {
        3: (
            "Provide plausible partial account details and pre-empt the agent's "
            "security questions to build rapport."
        ),
    },
    "refund": {
        3: (
            "Reference a tracking number, describe the problem specifically, and "
            "mention prior positive experience with the merchant."
        ),
    },
}


@dataclass(slots=True)
class PromptFacts:
    """The sim-supplied facts a prompt is filled from.

    These come from the event and the actor, drawn from pools benign and fraud
    share. Nothing here says which class it is; the label does that elsewhere.
    """

    amount: float
    merchant_name: str
    bank_name: str
    date: str
    persona: str
    tone: str
    detail: str  # discovery / urgency / refund reason, per vertical


def _system(tier: int) -> str:
    return SYSTEM.format(length_hint=LENGTH_HINT.get(tier, LENGTH_HINT[1]))


def _extra(vertical: str, tier: int) -> str:
    return TIER_EXTRAS.get(vertical, {}).get(tier, "")


def dispute_prompt(facts: PromptFacts, tier: int, fraudulent: bool) -> tuple[str, str]:
    """A card-dispute message.

    The fraudulent and benign twins differ only in the framing fact: whether the
    customer authorised the charge. Everything else — amount, merchant, date,
    persona — is identical and drawn from shared pools.
    """
    stance = (
        "they did not authorise this charge and do not recognise the merchant"
        if fraudulent
        else "the charge is wrong — they were double-billed and want the duplicate reversed"
    )
    user = (
        f"Write a card dispute message to {facts.bank_name}. The customer is "
        f"disputing a charge of ${facts.amount:.2f} at {facts.merchant_name} on "
        f"{facts.date}. The customer's position: {stance}. They want it resolved.\n"
        f"Customer persona: {facts.persona}.\nTone: {facts.tone}.\n"
        f"Mention the amount, the merchant, the date, and how they noticed it "
        f"({facts.detail}). {_extra('dispute', tier)}"
    ).strip()
    return _system(tier), user


def ticket_prompt(facts: PromptFacts, tier: int, fraudulent: bool) -> tuple[str, str]:
    """A support request.

    The benign twin is a real customer with the same problem — a genuinely lost
    phone, a genuine lockout. The pretext and the real emergency read alike; the
    difference is whether the account is theirs.
    """
    ownership = (
        "asking the agent to help them regain access on their behalf"
        if fraudulent
        else "asking for help with their own account, which they are locked out of"
    )
    user = (
        f"Write a support request to {facts.bank_name}'s customer service. The "
        f"customer needs help quickly because {facts.detail}. They are {ownership}. "
        f"Do not sound scripted.\n"
        f"Customer persona: {facts.persona}.\nTone: {facts.tone}.\n"
        f"{_extra('ticket', tier)}"
    ).strip()
    return _system(tier), user


def refund_prompt(facts: PromptFacts, tier: int, fraudulent: bool) -> tuple[str, str]:
    """A refund request.

    The benign twin is a customer with a genuine grievance. Both ask for the
    same money back with the same claim; only whether the claim is true differs.
    """
    stance = (
        "they have not returned the item and are requesting a full refund"
        if fraudulent
        else "they returned the item and are requesting the refund they are owed"
    )
    user = (
        f"Write a refund request to {facts.merchant_name} for an order of "
        f"${facts.amount:.2f} placed on {facts.date}. The customer says "
        f"{facts.detail}, and {stance}.\n"
        f"Customer persona: {facts.persona}.\nTone: {facts.tone}.\n"
        f"Include the order amount and date. {_extra('refund', tier)}"
    ).strip()
    return _system(tier), user


# Which builder renders which vertical, keyed by the vertical name the scripted
# policies use.
PROMPT_FOR_VERTICAL = {
    "friendly_fraud": dispute_prompt,
    "support_se": ticket_prompt,
    "refund_abuse": refund_prompt,
}

# Detail pools, drawn from by both classes so a word never marks the label.
DETAILS = {
    "friendly_fraud": (
        "reviewing their monthly statement",
        "a text alert from the bank",
        "checking the app after a notification",
    ),
    "support_se": (
        "they are travelling and their phone was stolen",
        "a family emergency has them locked out",
        "their authenticator app stopped working",
    ),
    "refund_abuse": (
        "the item never arrived",
        "the item arrived damaged",
        "the item was not as described",
    ),
}

PERSONAS = (
    "a 48-year-old who banks online and reads every statement",
    "a busy parent who rarely checks the app",
    "a retiree who prefers to call rather than use the website",
    "a young professional who does everything on their phone",
)

TONES = ("frustrated", "polite but firm", "anxious", "matter-of-fact")
