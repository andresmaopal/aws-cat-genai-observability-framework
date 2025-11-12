#!/usr/bin/env python3
import json
import boto3
import re
from typing import Dict, Set

def is_text_model(model_id: str, model_name: str = "") -> bool:
    """Check if model is a text generation model (not image/embedding)"""
    exclude_patterns = [
        'embed', 'image', 'stable', 'titan-image', 'nova-canvas', 'nova-reel', 
        'sd3', 'diffusion', 'inpaint', 'background', 'recolor', 'replace', 
        'erase', 'sketch', 'structure', 'style', 'ultra', 'core'
    ]
    model_lower = (model_id + " " + model_name).lower()
    return not any(pattern in model_lower for pattern in exclude_patterns)

def generate_short_name(model_id: str) -> str:
    """Generate short name from model ID"""
    # Handle global profiles
    if model_id.startswith('global.'):
        model_id = model_id.replace('global.', '')
    
    # Extract provider and model parts
    if '.' in model_id:
        parts = model_id.split('.')
        if len(parts) >= 2:
            model_part = parts[1]
            # Handle specific patterns for better naming
            if 'claude-sonnet-4-5' in model_part:
                return 'claude-4-5-sonnet'
            elif 'claude-opus-4-1' in model_part:
                return 'claude-4-1-opus'
            elif 'claude-3-7-sonnet' in model_part:
                return 'claude-3-7-sonnet'
            # Remove version suffixes and timestamps
            clean_name = re.sub(r'-v?\d+.*$', '', model_part)
            clean_name = re.sub(r'-\d{8}.*$', '', clean_name)
            return clean_name
    return model_id.replace(':', '-').replace('.', '-')

def get_bedrock_models(region: str) -> Dict[str, Dict]:
    """Get available models from Bedrock in specified region"""
    client = boto3.client('bedrock', region_name=region)
    models = {}
    
    try:
        # Get inference profiles
        profiles = client.list_inference_profiles()
        for profile in profiles.get('inferenceProfileSummaries', []):
            model_id = profile['inferenceProfileId']
            if is_text_model(model_id):
                short_name = generate_short_name(model_id)
                models[short_name] = {
                    "model_id": model_id,
                    "region_name": region,
                    "temperature": 0.2,
                    "inference_type": "INFERENCE_PROFILE",
                    "tooling_enabled": True
                }
    except Exception as e:
        print(f"Error getting inference profiles in {region}: {e}")
    
    try:
        # Get foundation models
        foundation_models = client.list_foundation_models()
        for model in foundation_models.get('modelSummaries', []):
            model_id = model['modelId']
            model_name = model.get('modelName', '')
            if is_text_model(model_id, model_name):
                short_name = generate_short_name(model_id)
                if short_name not in models:  # Don't override inference profiles
                    models[short_name] = {
                        "model_id": model_id,
                        "region_name": region,
                        "temperature": 0.2,
                        "inference_type": "ON_DEMAND",
                        "tooling_enabled": True
                    }
    except Exception as e:
        print(f"Error getting foundation models in {region}: {e}")
    
    return models

def sync_models():
    """Sync model_list.json with available Bedrock models"""
    # Load existing models
    try:
        with open('model_list.json', 'r') as f:
            existing_models = json.load(f)
    except FileNotFoundError:
        existing_models = {}
    
    # Get existing model IDs to avoid duplicates
    existing_model_ids: Set[str] = {config['model_id'] for config in existing_models.values()}
    
    # Get models from both regions
    all_models = {}
    for region in ['us-east-1', 'us-west-2']:
        print(f"Fetching models from {region}...")
        region_models = get_bedrock_models(region)
        
        for short_name, config in region_models.items():
            # Skip if model ID already exists
            if config['model_id'] in existing_model_ids:
                continue
                
            # Handle name conflicts by appending region
            final_name = short_name
            if final_name in all_models or final_name in existing_models:
                final_name = f"{short_name}-{region.replace('us-', '')}"
            
            all_models[final_name] = config
    
    # Merge with existing models
    updated_models = {**existing_models, **all_models}
    
    # Write updated model list
    with open('model_list.json', 'w') as f:
        json.dump(updated_models, f, indent=2)
    
    print(f"Added {len(all_models)} new models to model_list.json")
    print(f"Total models: {len(updated_models)}")
    
    if all_models:
        print("\nNew models added:")
        for name in sorted(all_models.keys()):
            print(f"  - {name}")

if __name__ == "__main__":
    sync_models()
