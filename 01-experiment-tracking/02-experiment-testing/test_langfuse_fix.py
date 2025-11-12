#!/usr/bin/env python3
"""
Test script to verify Langfuse V3 fix
"""

import os
import yaml
import uuid
from utils_litellm_fixed import UnifiedTester

# Set up environment variables
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-f8c36d47-cf05-46e2-9c32-28a592d59eae"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9ccabb24-56d9-4c29-94c4-c4b0e391f54b"
os.environ["LANGFUSE_HOST"] = "http://langfu-loadb-ukoqudmq8a8v-2110705221.us-east-1.elb.amazonaws.com"

def main():
    """Test the fixed Langfuse V3 integration"""
    
    print("🚀 Testing Fixed Langfuse V3 Integration\n")
    
    # Initialize tester
    tester = UnifiedTester()
    
    # Load configuration
    with open('config_experiments.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    prompts = config['system_prompts']
    test_queries = config['test_queries']
    
    # Create trace attributes
    session_id = str(uuid.uuid4())
    trace_attributes = {
        "operation.name": "Langfuse V3 Test",
        "langfuse.trace.name": "Fixed Langfuse V3 Integration Test",
        "session.id": session_id,
        "user.id": "test@example.com",
        "langfuse.tags": ["test", "langfuse-v3", "fixed"],
        "langfuse.environment": "development"
    }
    
    # Run a simple test
    results = tester.run_test(
        models=["bedrock/us.amazon.nova-micro-v1:0"],
        system_prompts=["version1"],
        queries=test_queries[0],
        prompts_dict=prompts,
        tool=None,  # No tools for this test
        trace_attributes=trace_attributes,
        save_to_csv=True
    )
    
    # Display results
    tester.display_results(results)
    
    print(f"\n✅ Test completed! Check Langfuse dashboard for trace: {session_id}")

if __name__ == "__main__":
    main()
