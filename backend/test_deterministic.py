"""
Test script to verify deterministic behavior
Run this to confirm same input gives same output every time
"""

from openai_integration import openai_integration, simple_openai

print("=" * 60)
print("TESTING DETERMINISTIC BEHAVIOR")
print("=" * 60)
print()

# Test 1: Same prompt 3 times
print("Test: Calling OpenAI 3 times with SAME prompt")
print("Expected: All 3 results should be IDENTICAL")
print()

prompt = "Categorize this transaction: Salary payment to employees"

print("Call 1...")
result1 = simple_openai(prompt, max_tokens=30)
print(f"Result 1: {result1}")
print()

print("Call 2...")
result2 = simple_openai(prompt, max_tokens=30)
print(f"Result 2: {result2}")
print()

print("Call 3...")
result3 = simple_openai(prompt, max_tokens=30)
print(f"Result 3: {result3}")
print()

print("=" * 60)
if result1 == result2 == result3:
    print("SUCCESS: All 3 results are IDENTICAL!")
    print("Deterministic behavior is working correctly.")
else:
    print("FAIL: Results are DIFFERENT!")
    print("Deterministic behavior is NOT working.")
    print()
    print("Comparison:")
    print(f"Result 1 == Result 2: {result1 == result2}")
    print(f"Result 2 == Result 3: {result2 == result3}")
    print(f"Result 1 == Result 3: {result1 == result3}")
print("=" * 60)
