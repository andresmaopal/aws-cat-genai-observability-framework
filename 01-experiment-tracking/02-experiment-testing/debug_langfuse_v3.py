#!/usr/bin/env python3
"""
Debug script for Langfuse V3 tracing issues
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
    """Test direct Langfuse V3 connection"""
    print("🔍 Testing direct Langfuse V3 connection...")
    
    try:
        from langfuse import Langfuse
        
        # Initialize Langfuse client
        langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ["LANGFUSE_HOST"]
        )
        
        # Create a simple trace
        trace = langfuse.trace(
            name="Debug Test Trace",
            input={"test": "direct connection"},
            metadata={"test_type": "debug", "version": "v3"}
        )
        
        # Create a span
        span = trace.span(
            name="Test Span",
            input={"message": "Hello Langfuse V3"}
        )
        
        # End span and trace
        span.end(output={"result": "success"})
        trace.update(output={"status": "completed"})
        
        # Flush to ensure data is sent
        langfuse.flush()
        
        print(f"✅ Direct Langfuse V3 test successful! Trace ID: {trace.id}")
        return True
        
    except Exception as e:
        print(f"❌ Direct Langfuse V3 test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_opentelemetry_setup():
    """Test OpenTelemetry setup for Langfuse V3"""
    print("\n🔍 Testing OpenTelemetry setup...")
    
    try:
        # Set up OTEL environment
        otel_endpoint = f"{os.environ['LANGFUSE_HOST']}/api/public/otel/v1/traces"
        auth_token = base64.b64encode(
            f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
        ).decode()
        
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_token}"
        
        print(f"✅ OTEL Endpoint: {otel_endpoint}")
        print(f"✅ OTEL Headers configured")
        
        # Test OTEL tracing
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        
        # Set up tracer
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Set up OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=otel_endpoint,
            headers={"Authorization": f"Basic {auth_token}"}
        )
        
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Create a test span
        with tracer.start_as_current_span("otel_test_span") as span:
            span.set_attribute("test.type", "opentelemetry")
            span.set_attribute("test.timestamp", datetime.now().isoformat())
            span.add_event("Test event", {"message": "OTEL test successful"})
        
        # Force flush
        trace.get_tracer_provider().force_flush(timeout_millis=5000)
        
        print("✅ OpenTelemetry test successful!")
        return True
        
    except Exception as e:
        print(f"❌ OpenTelemetry test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_strands_agent_tracing():
    """Test Strands Agent with Langfuse V3 tracing"""
    print("\n🔍 Testing Strands Agent with Langfuse V3...")
    
    try:
        from strands import Agent
        from strands.models.litellm import LiteLLMModel
        
        # Create LiteLLM model
        model = LiteLLMModel(
            model="bedrock/us.amazon.nova-micro-v1:0",
            region_name="us-east-1"
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
        
        # Test simple query
        response = agent.run("Say hello and confirm tracing is working")
        
        print(f"✅ Strands Agent test successful!")
        print(f"Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Strands Agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🚀 Starting Langfuse V3 Debug Tests\n")
    
    results = []
    
    # Test 1: Direct Langfuse connection
    results.append(("Direct Langfuse V3", test_langfuse_v3_direct()))
    
    # Test 2: OpenTelemetry setup
    results.append(("OpenTelemetry Setup", test_opentelemetry_setup()))
    
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
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
