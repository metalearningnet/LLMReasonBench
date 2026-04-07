COT_TOKENS = {
    'memory': {
        'description': 'Extract and state ONLY the given facts, numbers, or formulas from the problem statement.',
        'prerequisite': True # Generate these steps first and use as context for the rest
    },
    'reason': {
        'description': 'Explain the reasoning process to derive the answer, using facts from the prerequisite memory steps and logic.'
    }
}

END_MARK = True # True: <token> content </token> | False: <token>: content
