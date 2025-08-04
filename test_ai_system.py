#!/usr/bin/env python3
"""
Demo script to test our AI-powered docstring generator.
This script will demonstrate the complete system in action!
"""

import sys
import os
from src.docstring_generator import DocstringGenerator

def load_test_functions():
    """Load the test functions from our test file."""
    with open('tests/test_functions.py', 'r') as f:
        return f.read()

def test_ai_docstring_generator():
    """Test our documentation system on various functions."""
    
    print("🤖 Automated Docstring Generator Demo")
    print("=" * 50)
    
    # Initialize your documentation system
    print("\n📥 Loading documentation engine (this may take a moment)...")
    try:
        generator = DocstringGenerator()
    except Exception as e:
        print(f"❌ Failed to load documentation engine: {e}")
        print("💡 Make sure you've installed all dependencies: pip install -r requirements.txt")
        return
    
    # Load test functions
    print("\n📋 Loading test functions...")
    source_code = load_test_functions()
    
    # List of functions to test
    test_functions = [
        "calculate_bmi",
        "fibonacci_sequence", 
        "merge_sorted_arrays",
        "validate_email_format",
        "process_user_data"
    ]
    
    print(f"\n🧪 Testing system on {len(test_functions)} functions:")
    print("-" * 50)
    
    # Test each function
    for func_name in test_functions:
        print(f"\n🔍 Testing function: {func_name}")
        print("-" * 30)
        
        # Generate docstring using your AI system
        docstring = generator.generate_docstring(source_code, func_name)
        
        if docstring:
            print("✅ Generated docstring:")
            print(docstring)
        else:
            print("❌ Failed to generate docstring")
        
        print("\n" + "="*50)
    
    print("\n🎉 Demo completed! Your automated documentation assistant is working!")

if __name__ == "__main__":
    test_ai_docstring_generator()