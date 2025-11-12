#!/usr/bin/env python3
"""
Fix for LiteLLM Bedrock model tokenization error
"""

# Enable LiteLLM debug mode to see detailed error information
import litellm
litellm._turn_on_debug()

# Alternative model configurations that should work
WORKING_MODELS = [
    "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # This one works
    "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0",    # This one works
    "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # Alternative
    "bedrock/us.meta.llama3-2-90b-instruct-v1:0",           # Alternative
]

# Problematic model (causing the error)
PROBLEMATIC_MODEL = "bedrock/openai.gpt-oss-120b-1:0"

print("🔧 Model Configuration Fix")
print("=" * 50)
print(f"❌ Problematic model: {PROBLEMATIC_MODEL}")
print("✅ Working alternatives:")
for model in WORKING_MODELS:
    print(f"   - {model}")
