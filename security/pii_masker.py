import re

def mask_pii(text: str) -> str:
    """
    Masks Personally Identifiable Information (PII) such as emails and phone numbers.
    """
    # Mask emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[EMAIL_REDACTED]', text)
    
    # Mask phone numbers (simple pattern, can be improved)
    phone_pattern = r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    # Refining phone pattern to avoid matching random numbers like dates or versions
    refined_phone_pattern = r'(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
    text = re.sub(refined_phone_pattern, '[PHONE_REDACTED]', text)
    
    # Mask addresses, SSN, etc. can be added here if needed.
    return text

def sanitize_input(text: str) -> str:
    """
    Sanitizes input text to prevent prompt injection attacks.
    Removes system instruction patterns.
    """
    # Strip potential XML/HTML tags often used for prompt injection
    text = re.sub(r'<system>.*?</system>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<instruction>.*?</instruction>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL) # remove code blocks that might contain injections
    
    # Optional: Truncate excessively long inputs
    max_length = 50000 
    if len(text) > max_length:
        text = text[:max_length]
        
    return text
