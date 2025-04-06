import re

def lexical_analyzer(code):
    """Simple lexical analyzer for a subset of a programming language"""
    # Define token patterns
    token_patterns = [
        ('KEYWORD', r'\b(if|else|while|for|int|float|return)\b'),
        ('IDENTIFIER', r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'),
        ('NUMBER', r'\b\d+(\.\d+)?\b'),
        ('OPERATOR', r'[+\-*/=<>]'),
        ('DELIMITERS', r'[(){};,]'),
        ('WHITESPACE', r'\s+'),
        ('COMMENT', r'//.*')
    ]
    
    # Combine patterns
    regex_pattern = '|'.join('(?P<%s>%s)' % (name, pattern) for name, pattern in token_patterns)
    
    # Tokenize the input
    tokens = []
    position = 0
    
    print("Lexical Analysis Results:")
    print("-----------------------")
    print(f"Input: {code}")
    print("\nTokens:")
    
    for match in re.finditer(regex_pattern, code):
        kind = match.lastgroup
        value = match.group()
        position = match.start()
        
        if kind == 'WHITESPACE' or kind == 'COMMENT':
            continue  # Skip whitespace and comments
        
        tokens.append((kind, value))
        print(f"  {kind}: '{value}'")
    
    return tokens

# Example usage
sample_code = """
if (x > 5) {
    y = x + 10;
    // This is a comment
    return y;
}
"""

tokens = lexical_analyzer(sample_code)