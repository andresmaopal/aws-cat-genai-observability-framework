# Langfuse V3 Tracing Fix

## Issues Identified

1. **Incorrect Langfuse V3 API usage** - The original code used non-existent methods like `langfuse.trace()` and `langfuse.create_trace()`
2. **Wrong OpenTelemetry context management** - Missing proper span context handling
3. **Strands Agent method calls** - Using `agent.run()` instead of `agent.invoke_async()`

## Solutions Applied

### 1. Fixed Langfuse V3 API Usage

**Before (Incorrect):**
```python
trace = langfuse.trace(name="test")  # ❌ Method doesn't exist
```

**After (Correct):**
```python
# Create main span (this creates the trace automatically)
main_trace = langfuse_client.start_span(
    name=trace_name,
    input={"test_type": "litellm_unified_test"},
    metadata={"environment": "development"}
)

# Update trace
langfuse_client.update_current_trace(output={"status": "completed"})

# Score trace
langfuse_client.score_current_trace(
    name="success_rate", 
    value=0.95
)
```

### 2. Fixed Strands Agent Usage

**Before (Incorrect):**
```python
response = agent.run("query")  # ❌ Method doesn't exist
```

**After (Correct):**
```python
async def run_test():
    return await agent.invoke_async("query")

response = asyncio.run(run_test())
```

### 3. Proper OpenTelemetry Integration

```python
# Set up OTEL environment for Langfuse V3
otel_endpoint = f"{langfuse_host}/api/public/otel/v1/traces"
auth_token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_token}"
```

## How to Apply the Fix

### Option 1: Replace utils_litellm.py
```bash
cp utils_litellm_fixed.py utils_litellm.py
```

### Option 2: Update your notebook cells

Replace the LangFuse setup cell with:

```python
import os, base64

# Set environment variables
os.environ["LANGFUSE_SECRET_KEY"] = "your-secret-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = "your-public-key" 
os.environ["LANGFUSE_HOST"] = "your-langfuse-host"

# Set up OpenTelemetry endpoint
otel_endpoint = f"{os.environ['LANGFUSE_HOST']}/api/public/otel/v1/traces"
auth_token = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_token}"

print("✅ Langfuse V3 environment configured")
```

## Test Results

✅ **Direct Langfuse V3 connection**: Working
✅ **OpenTelemetry integration**: Working  
✅ **Strands Agent tracing**: Working
✅ **End-to-end tracing**: Working

The fix resolves all Langfuse V3 tracing issues and enables proper observability in your notebook.
