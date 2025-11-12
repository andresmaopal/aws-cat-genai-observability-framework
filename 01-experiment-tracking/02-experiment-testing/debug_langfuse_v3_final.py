#!/usr/bin/env python3
"""
Final corrected debug script for Langfuse V3 tracing issues
"""

import os
import base64
import uuid
from datetime import datetime

# Set up environment variables
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-f8c36d47-cf05-46e2-9c32-28a592d59eae"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9ccabb24-56d9-4c29-94c4-c4b0e391f54b"
os.environ["LANGFUSE_HOST"] = "http://langfu-loadb-ukoqudmq8a8v-2110705221.us-east-1.elb.amazonaws.com"

def test_langfuse_v3_direct():
    """Test direct Langfuse V3 connection with correct API"""
    print("🔍 Testing direct Langfuse V3 connection...")
    
    try:
        from langfuse import Langfuse
        
        # Initialize Langfuse client
        langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ["LANGFUSE_HOST"]
        )
        
        # Create a trace using correct V3 API
        trace_id = langfuse.create_trace_id()
        
        # Start a span (which creates the trace)
        span = langfuse.start_span(
            name="Debug Test Trace",
            input={"test": "direct connection"},
            metadata={"test_type": "debug", "version": "v3"},
            trace_id=trace_id
        )
        
        # Create a child span
        child_span = langfuse.start_span(
            name="Test Child Span",
            input={"message": "Hello Langfuse V3"},
            parent_observation_id=span.id,
            trace_id=trace_id
        )
        
        # Update spans
        langfuse.update_current_span(
            output={"result": "child success"}
        )
        
        langfuse.update_current_trace(
            output={"status": "completed"}
        )
        
        # Flush to ensure data is sent
        langfuse.flush()
        
        print(f"✅ Direct Langfuse V3 test successful! Trace ID: {trace_id}")
        return True
        
    except Exception as e:
        print(f"❌ Direct Langfuse V3 test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_strands_agent_tracing():
    """Test Strands Agent with Langfuse V3 tracing using correct API"""
    print("\n🔍 Testing Strands Agent with Langfuse V3...")
    
    try:
        from strands import Agent
        from strands.models.litellm import LiteLLMModel
        
        # Create LiteLLM model with correct parameters
        model = LiteLLMModel(
            model_id="bedrock/us.amazon.nova-micro-v1:0"
        )
        
        # Create agent with trace attributes
        trace_attributes = {
            "operation.name": "Debug Test",
            "langfuse.trace.name": "Strands Agent Debug",
            "session.id": str(uuid.uuid4()),
            "user.id": "debug@test.com",
            "langfuse.tags": ["debug", "test"],
            "langfuse.environment": "development"
        }
        
        agent = Agent(
            model=model,
            system_prompt="You are a helpful assistant for debugging.",
            trace_attributes=trace_attributes
        )
        
        # Test simple query using correct async method
        import asyncio
        
        async def test_agent():
            response = await agent.invoke_async("Say hello and confirm tracing is working")
            return response
        
        response = asyncio.run(test_agent())
        
        print(f"✅ Strands Agent test successful!")
        print(f"Response: {response.content[:100] if hasattr(response, 'content') else str(response)[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Strands Agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_opentelemetry_integration():
    """Test OpenTelemetry integration with Langfuse V3"""
    print("\n🔍 Testing OpenTelemetry integration...")
    
    try:
        # Set up OTEL environment for Langfuse
        otel_endpoint = f"{os.environ['LANGFUSE_HOST']}/api/public/otel/v1/traces"
        auth_token = base64.b64encode(
            f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
        ).decode()
        
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_token}"
        
        print(f"✅ OTEL Endpoint: {otel_endpoint}")
        print(f"✅ OTEL Headers configured")
        
        # Test Langfuse OTEL integration
        from langfuse import Langfuse
        
        langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ["LANGFUSE_HOST"]
        )
        
        # Use Langfuse's OTEL integration
        with langfuse.start_as_current_span("otel_integration_test") as span:
            span.set_attribute("test.type", "otel_integration")
            span.set_attribute("test.timestamp", datetime.now().isoformat())
            span.add_event("OTEL integration test", {"message": "Testing Langfuse OTEL"})
        
        langfuse.flush()
        
        print("✅ OpenTelemetry integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ OpenTelemetry integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🚀 Starting Langfuse V3 Debug Tests (Final)\n")
    
    results = []
    
    # Test 1: Direct Langfuse connection
    results.append(("Direct Langfuse V3", test_langfuse_v3_direct()))
    
    # Test 2: OpenTelemetry integration
    results.append(("OpenTelemetry Integration", test_opentelemetry_integration()))
    
    # Test 3: Strands Agent tracing
    results.append(("Strands Agent Tracing", test_strands_agent_tracing()))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 DEBUG RESULTS SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} | {status}")
    
    total_passed = sum(results[i][1] for i in range(len(results)))
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All tests passed! Langfuse V3 should be working.")
        print("\n💡 RECOMMENDATIONS:")
        print("1. ✅ Langfuse V3 connection is working")
        print("2. ✅ OpenTelemetry integration is configured")
        print("3. ✅ Strands Agent tracing should work with your notebook")
    else:
        print("⚠️  Some tests failed. Issues identified:")
        if not results[0][1]:
            print("- Langfuse V3 API connection issue")
        if not results[1][1]:
            print("- OpenTelemetry configuration issue")
        if not results[2][1]:
            print("- Strands Agent integration issue")

if __name__ == "__main__":
    main()
