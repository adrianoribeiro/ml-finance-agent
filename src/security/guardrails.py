"""Input and output guardrails for the credit risk agent."""
import logging
import re

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 4096

# Common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system:\s*",
    r"<\|?im_start\|?>",
    r"\[INST\]",
    r"forget\s+(everything|all|your\s+instructions)",
    r"act\s+as\s+(if|a)",
    r"do\s+not\s+follow\s+your\s+(rules|instructions)",
]

# Topics the agent should not discuss
OFF_TOPIC_PATTERNS = [
    r"(hack|exploit|attack|bypass)",
    r"(password|credential|secret\s+key)",
    r"(sql\s+injection|xss|csrf)",
]


class InputGuardrail:
    def __init__(self, allowed_topics: list[str] | None = None):
        self.allowed_topics = allowed_topics or []
        self._injection_re = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]
        self._offtopic_re = [
            re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Validate user input. Returns (is_valid, reason)."""
        # Check length
        if len(user_input) > MAX_INPUT_LENGTH:
            logger.warning(f"Input too long: {len(user_input)} chars")
            return False, f"Input exceeds maximum length ({MAX_INPUT_LENGTH} chars)."

        if not user_input.strip():
            return False, "Input is empty."

        # Check prompt injection
        for pattern in self._injection_re:
            if pattern.search(user_input):
                logger.warning(f"Prompt injection detected: {user_input[:100]}")
                return False, "Suspicious input detected."

        # Check off-topic
        for pattern in self._offtopic_re:
            if pattern.search(user_input):
                logger.warning(f"Off-topic input: {user_input[:100]}")
                return False, "This topic is outside the scope of credit risk analysis."

        return True, "OK"


class OutputGuardrail:
    def __init__(self):
        # PII patterns for Brazilian data
        self._pii_patterns = {
            "cpf": re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"),
            "phone": re.compile(r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}"),
            "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        }

    def sanitize(self, output: str) -> str:
        """Remove PII from agent output."""
        sanitized = output
        for pii_type, pattern in self._pii_patterns.items():
            matches = pattern.findall(sanitized)
            if matches:
                logger.warning(f"PII detected in output ({pii_type}): {len(matches)} matches")
                sanitized = pattern.sub(f"[{pii_type.upper()}_REDACTED]", sanitized)
        return sanitized
