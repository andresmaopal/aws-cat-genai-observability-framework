#!/usr/bin/env python3
import json
import boto3

def get_bedrock_models():
    """Get all text models from Bedrock in both regions"""
    models = {}
    
    for region in ['us-east-1', 'us-west-2']:
        client = boto3.client('bedrock', region_name=region)
        response = client.list_foundation_models(byOutputModality='TEXT')
        
        for model in response['modelSummaries']:
            if model['modelLifecycle']['status'] == 'ACTIVE':
                model_id = model['modelId']
                
                # Determine inference type
                inference_types = model.get('inferenceTypesSupported', [])
                if 'INFERENCE_PROFILE' in inference_types:
                    inference_type = 'INFERENCE_PROFILE'
                elif 'ON_DEMAND' in inference_types:
                    inference_type = 'ON_DEMAND'
                else:
                    continue
                
                # Create friendly name
                provider = model['providerName'].lower()
                model_name = model['modelName'].lower()
                
                if 'claude' in model_name:
                    if 'sonnet 4' in model_name:
                        friendly_name = 'claude-4-sonnet'
                    elif 'opus 4.1' in model_name:
                        friendly_name = 'claude-4.1-opus'
                    elif 'opus 4' in model_name:
                        friendly_name = 'claude-4-opus'
                    elif '3.7 sonnet' in model_name:
                        friendly_name = 'claude-3.7-sonnet'
                    elif '3.5 sonnet v2' in model_name:
                        friendly_name = 'claude-3.5-sonnet-v2'
                    elif '3.5 sonnet' in model_name:
                        friendly_name = 'claude-3.5-sonnet'
                    elif '3.5 haiku' in model_name:
                        friendly_name = 'claude-3.5-haiku'
                    elif '3 opus' in model_name:
                        friendly_name = 'claude-3-opus'
                    elif '3 sonnet' in model_name:
                        friendly_name = 'claude-3-sonnet'
                    elif '3 haiku' in model_name:
                        friendly_name = 'claude-3-haiku'
                    else:
                        continue
                elif 'nova' in model_name:
                    if 'premier' in model_name:
                        friendly_name = 'nova-premier'
                    elif 'pro' in model_name:
                        friendly_name = 'nova-pro'
                    elif 'lite' in model_name:
                        friendly_name = 'nova-lite'
                    elif 'micro' in model_name:
                        friendly_name = 'nova-micro'
                    else:
                        continue
                elif 'llama' in model_name:
                    if '4 scout' in model_name:
                        friendly_name = 'llama-4-scout'
                    elif '4 maverick' in model_name:
                        friendly_name = 'llama-4-maverick'
                    elif '3.3 70b' in model_name:
                        friendly_name = 'llama-3.3-70b'
                    elif '3.2 90b' in model_name:
                        friendly_name = 'llama-3.2-90b'
                    elif '3.2 11b' in model_name:
                        friendly_name = 'llama-3.2-11b'
                    elif '3.2 3b' in model_name:
                        friendly_name = 'llama-3.2-3b'
                    elif '3.2 1b' in model_name:
                        friendly_name = 'llama-3.2-1b'
                    elif '3.1 70b' in model_name:
                        friendly_name = 'llama-3.1-70b'
                    elif '3.1 8b' in model_name:
                        friendly_name = 'llama-3.1-8b'
                    elif '3.1 405b' in model_name:
                        friendly_name = 'llama-3.1-405b'
                    elif '3 70b' in model_name:
                        friendly_name = 'llama-3-70b'
                    elif '3 8b' in model_name:
                        friendly_name = 'llama-3-8b'
                    else:
                        continue
                elif 'palmyra' in model_name:
                    if 'x5' in model_name:
                        friendly_name = 'palmyra-x5'
                    elif 'x4' in model_name:
                        friendly_name = 'palmyra-x4'
                    else:
                        continue
                else:
                    # Skip other models for now
                    continue
                
                # Set temperature based on model type
                if 'haiku' in friendly_name or 'micro' in friendly_name or '1b' in friendly_name:
                    temp = 0.7
                elif 'lite' in friendly_name or '3b' in friendly_name or '8b' in friendly_name:
                    temp = 0.6
                elif 'pro' in friendly_name or 'sonnet' in friendly_name:
                    temp = 0.5
                elif 'premier' in friendly_name or 'opus' in friendly_name or '405b' in friendly_name:
                    temp = 0.3
                else:
                    temp = 0.5
                
                models[friendly_name] = {
                    "model_id": model_id,
                    "temperature": temp,
                    "region_name": region,
                    "inference_type": inference_type
                }
    
    return models

def sync_model_list():
    """Sync model list with latest Bedrock models"""
    
    # Get current models
    try:
        with open('model_list.json', 'r') as f:
            current_models = json.load(f)
    except FileNotFoundError:
        current_models = {}
    
    # Get latest Bedrock models
    bedrock_models = get_bedrock_models()
    
    # Add missing models
    added_count = 0
    for name, config in bedrock_models.items():
        if name not in current_models:
            current_models[name] = config
            current_models[name]["tooling_enabled"] = "UNKNOWN"
            added_count += 1
            print(f"Added: {name}")
    
    # Save updated list
    with open('model_list.json', 'w') as f:
        json.dump(current_models, f, indent=4)
    
    print(f"\nSync complete! Added {added_count} new models.")
    return added_count > 0

if __name__ == "__main__":
    if sync_model_list():
        print("Running tooling tests for new models...")
        import test_tooling
        test_tooling.update_model_list()
    else:
        print("No new models found.")
