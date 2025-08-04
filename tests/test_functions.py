"""
Test functions without docstrings for AI documentation generation.
These functions will be used to validate our AI system.
"""

def calculate_bmi(weight, height):
    if height <= 0:
        raise ValueError("Height must be positive")
    return weight / (height ** 2)

def fibonacci_sequence(n: int, memo: dict = None) -> int:
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_sequence(n-1, memo) + fibonacci_sequence(n-2, memo)
    return memo[n]

def merge_sorted_arrays(arr1, arr2):
    result = []
    i, j = 0, 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result

def validate_email_format(email: str) -> bool:
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not email or not isinstance(email, str):
        return False
    return bool(re.match(pattern, email.strip()))

def process_user_data(users, filter_active=True, sort_by="name"):
    if not users:
        return []
    
    processed = users.copy()
    
    if filter_active:
        processed = [user for user in processed if user.get('active', True)]
    
    if sort_by in ['name', 'email', 'created_date']:
        processed.sort(key=lambda x: x.get(sort_by, ''))
    
    return processed