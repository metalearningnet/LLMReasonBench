MEMORY_TOKEN_NAME = 'memory'
REASON_TOKEN_NAME = 'reason'
COT_TOKEN_NAMES = [MEMORY_TOKEN_NAME, REASON_TOKEN_NAME]

STEPS = {
    MEMORY_TOKEN_NAME: {
        'description': "Extract and state ONLY the given facts, numbers, or formulas from the problem statement.",
        'guidelines': [
            "Restate exactly what is provided in the question",
            "Never include calculations, reasoning, or inferences",
            "Separate different facts into individual steps when possible"
        ]
    },
    REASON_TOKEN_NAME: {
        'description': "Perform calculations and logical operations using the facts from memory steps.",
        'guidelines': [
            "Reference facts from memory steps (e.g., 'Using that...')",
            "Show each calculation or logical step clearly",
            "Explain the reasoning process, not just the operation"
        ]
    }
}

MEMORY_TOKEN = f'<{MEMORY_TOKEN_NAME}>'
REASON_TOKEN = f'<{REASON_TOKEN_NAME}>'

COT_TOKENS = [MEMORY_TOKEN, REASON_TOKEN]
