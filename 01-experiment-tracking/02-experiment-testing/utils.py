#!/usr/bin/env python3
"""
AWS Strands Agents SDK - Unified Testing Framework

Unified class with run_test() method and human-readable results display.
"""

import time
import json
import os
import csv
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Union, Any, Optional, Tuple
import pandas as pd
from IPython.display import display

# Strands SDK imports
from strands import Agent
from strands.models import BedrockModel


class UnifiedTester:
    """Unified testing class with run_test() method and enhanced result display"""

    def __init__(self):
        self.models = self._load_models()

    def _load_models(self):
        """Load models from JSON file"""
        json_path = os.path.join(os.path.dirname(__file__), 'model_list.json')
        with open(json_path, 'r') as f:
            return json.load(f)

    def run_test(self, 
                 models: List[str], 
                 system_prompts: List[str], 
                 queries: Union[str, List[str]], 
                 prompts_dict: Dict[str, str], 
                 tool: List = None, 
                 trace: Any = None,
                 trace_attributes: Optional[Dict[str, Any]] = None,
                 save_to_csv: bool = True) -> List[Dict[str, Any]]:
        """
        Unified test method that allows testing multiple combinations of:
        - models: List of model IDs (e.g., ["claude-4-sonnet", "qwen3-235b"])
        - system_prompts: List of prompt versions (e.g., ["version1", "version2"])
        - queries: Single query string or list of queries
        - prompts_dict: Dictionary containing system prompts
        - tool: List of tools to use
        - save_to_csv: Whether to save results to CSV in test_results folder (default: True)
        
        Returns: List of test results with human-readable structure
        """
        # Normalize queries to list
        if isinstance(queries, str):
            queries = [queries]
        
        results = []
        total_tests = len(models) * len(system_prompts) * len(queries)
        current_test = 0
        
        print(f"\n🚀 Starting Unified Test Suite")
        print(f"📊 Total combinations to test: {total_tests}")
        print(f"🤖 Models: {models}")
        print(f"📝 Prompts: {system_prompts}")
        print(f"❓ Queries: {len(queries)} query(ies)")
        print("=" * 80)
        
        for model_name in models:
            for prompt_name in system_prompts:
                for query in queries:
                    current_test += 1
                    print(f"\n[{current_test}/{total_tests}] Testing: {model_name} | {prompt_name}")
                    print(f"Query: {query[:60]}{'...' if len(query) > 60 else ''}")
                    print("-" * 60)
                    
                    result = self._execute_single_test(
                        model_name, prompt_name, query, prompts_dict, tool, trace_attributes
                    )
                    results.append(result)
                    
                    # Brief status
                    status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                    print(f"{status} | Time: {result['response_time']:.2f}s")
                    
                    # Small delay between tests
                    time.sleep(0.5)
        
        print(f"\n🎉 Test Suite Completed! {len(results)} results generated.")
        
        # Auto-save to CSV if enabled
        if save_to_csv:
            self._save_results_to_csv(results)
        
        return results

    def _execute_single_test(self, model_name: str, prompt_name: str, query: str, 
                           prompts_dict: Dict[str, str], tool: List = None, 
                           trace_attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a single test combination"""
        try:
            # Validate inputs
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            if prompt_name not in prompts_dict:
                raise ValueError(f"Prompt '{prompt_name}' not found")
            
            # Create agent
            agent = self._create_agent(model_name, prompt_name, prompts_dict, tool, trace_attributes)
            
            # Execute test
            start_time = time.time()
            response = agent(query)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract response text
            if hasattr(response, 'message') and 'content' in response.message:
                response_text = response.message['content']
            else:
                response_text = str(response)
            
            return {
                "test_id": f"{model_name}_{prompt_name}_{hash(query) % 10000}",
                "model": model_name,
                "prompt": prompt_name,
                "query": query,
                "response": response_text,
                "response_time": response_time,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat(),
                "model_config": self.models[model_name]
            }
            
        except Exception as e:
            return {
                "test_id": f"{model_name}_{prompt_name}_{hash(query) % 10000}",
                "model": model_name,
                "prompt": prompt_name,
                "query": query,
                "response": "",
                "response_time": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "model_config": self.models.get(model_name, {})
            }

    def _create_agent(self, model_name: str, prompt_name: str, 
                     prompts_dict: Dict[str, str], tool: List = None,
                     trace_attributes: Optional[Dict[str, Any]] = None) -> Agent:
        """Create an agent with specified configuration"""
        model_config = self.models[model_name]
        
        # Create model
        model = BedrockModel(
            model_id=model_config["model_id"],
            temperature=model_config["temperature"],
            region_name=model_config["region_name"]
        )
        
        # Get system prompt
        system_prompt = prompts_dict[prompt_name]
        
        # Create agent
        agent_kwargs = {
            "model": model,
            "system_prompt": system_prompt
        }
        
        if tool:
            agent_kwargs["tools"] = tool
            
        if trace_attributes:
            import copy
            # Create a deep copy to avoid modifying the original
            dynamic_trace_attributes = copy.deepcopy(trace_attributes)
            
            # Dynamically add model trace attribute
            if "langfuse.tags" in dynamic_trace_attributes:
                # Create new tags list with only current model tag
                original_tags = [tag for tag in dynamic_trace_attributes["langfuse.tags"] if not tag.startswith("Model-")]
                dynamic_trace_attributes["langfuse.tags"] = original_tags + [f"Model-{model_name}"]
            else:
                # Create tags list with model tag
                dynamic_trace_attributes["langfuse.tags"] = [f"Model-{model_name}"]
            
            agent_kwargs["trace_attributes"] = dynamic_trace_attributes
        
        return Agent(**agent_kwargs)

    def _save_results_to_csv(self, results: List[Dict[str, Any]]) -> None:
        """Save results to CSV in test_results folder"""
        # Create test_results directory if it doesn't exist
        test_results_dir = "test_results"
        os.makedirs(test_results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(test_results_dir, f"test_results_{timestamp}.csv")
        
        if not results:
            print("No results to save.")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_id', 'model', 'prompt', 'query', 'response', 'analysis_text',
                'response_time', 'success', 'error', 'timestamp'
            ])
            writer.writeheader()
            for result in results:
                response_text = str(result.get('response', ''))
                analysis_text = self._extract_analysis_text(response_text)
                
                writer.writerow({
                    'test_id': result.get('test_id', ''),
                    'model': result.get('model', ''),
                    'prompt': result.get('prompt', ''),
                    'query': result.get('query', ''),
                    'response': response_text,
                    'analysis_text': analysis_text,
                    'response_time': result.get('response_time', 0),
                    'success': result.get('success', False),
                    'error': result.get('error', ''),
                    'timestamp': result.get('timestamp', '')
                })
        
        print(f"📁 Results automatically saved to {filename}")

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Display results in human-readable format"""
        if not results:
            print("No results to display.")
            return
        
        # Summary statistics
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n📈 RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"Total Tests: {len(results)}")
        print(f"✅ Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"❌ Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        
        if successful:
            avg_time = sum(r['response_time'] for r in successful) / len(successful)
            min_time = min(r['response_time'] for r in successful)
            max_time = max(r['response_time'] for r in successful)
            print(f"⏱️  Response Times - Avg: {avg_time:.2f}s | Min: {min_time:.2f}s | Max: {max_time:.2f}s")
        
        # Group results by model and prompt
        by_model = defaultdict(list)
        by_prompt = defaultdict(list)
        
        for result in results:
            by_model[result['model']].append(result)
            by_prompt[result['prompt']].append(result)
        
        # Model performance
        print(f"\n🤖 MODEL PERFORMANCE")
        print(f"{'='*50}")
        for model, model_results in by_model.items():
            model_successful = [r for r in model_results if r['success']]
            success_rate = len(model_successful) / len(model_results) * 100
            avg_time = sum(r['response_time'] for r in model_successful) / len(model_successful) if model_successful else 0
            print(f"{model:20} | Success: {success_rate:5.1f}% | Avg Time: {avg_time:6.2f}s | Tests: {len(model_results)}")
        
        # Prompt performance
        print(f"\n📝 PROMPT PERFORMANCE")
        print(f"{'='*50}")
        for prompt, prompt_results in by_prompt.items():
            prompt_successful = [r for r in prompt_results if r['success']]
            success_rate = len(prompt_successful) / len(prompt_results) * 100
            avg_time = sum(r['response_time'] for r in prompt_successful) / len(prompt_successful) if prompt_successful else 0
            print(f"{prompt:20} | Success: {success_rate:5.1f}% | Avg Time: {avg_time:6.2f}s | Tests: {len(prompt_results)}")


    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results and provide insights"""
        if not results:
            return {}
        
        analysis = {
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r['success']]),
            "failed_tests": len([r for r in results if not r['success']]),
            "success_rate": len([r for r in results if r['success']]) / len(results) * 100,
            "models_tested": list(set(r['model'] for r in results)),
            "prompts_tested": list(set(r['prompt'] for r in results)),
            "queries_tested": len(set(r['query'] for r in results))
        }
        
        successful = [r for r in results if r['success']]
        if successful:
            response_times = [r['response_time'] for r in successful]
            analysis.update({
                "avg_response_time": sum(response_times) / len(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times)
            })
        
        # Model rankings
        model_performance = defaultdict(list)
        for result in results:
            model_performance[result['model']].append(result)
        
        model_rankings = []
        for model, model_results in model_performance.items():
            successful_model = [r for r in model_results if r['success']]
            success_rate = len(successful_model) / len(model_results) * 100
            avg_time = sum(r['response_time'] for r in successful_model) / len(successful_model) if successful_model else float('inf')
            
            model_rankings.append({
                "model": model,
                "success_rate": success_rate,
                "avg_response_time": avg_time,
                "total_tests": len(model_results)
            })
        
        # Sort by success rate, then by response time
        model_rankings.sort(key=lambda x: (-x['success_rate'], x['avg_response_time']))
        analysis["model_rankings"] = model_rankings
        
        # Display analysis
        print(f"🔍 PERFORMANCE ANALYSIS")
        print(f"{'='*50}")
        print(f"Overall Success Rate: {analysis['success_rate']:.1f}%")
        print(f"Models Tested: {len(analysis['models_tested'])}")
        print(f"Prompts Tested: {len(analysis['prompts_tested'])}")
        print(f"Unique Queries: {analysis['queries_tested']}")
        
        if 'avg_response_time' in analysis:
            print(f"Average Response Time: {analysis['avg_response_time']:.2f}s")
        
        print(f"\n🏆 MODEL RANKINGS (by success rate, then speed)")
        print(f"{'='*60}")
        for i, model_data in enumerate(model_rankings, 1):
            print(f"{i}. {model_data['model']:20} | {model_data['success_rate']:5.1f}% | {model_data['avg_response_time']:6.2f}s")
        
        return analysis

    def _extract_analysis_text(self, response: str) -> str:
        """Extract text from <analysis></analysis> tags"""
        import re
        if not isinstance(response, str):
            response = str(response)
        
        match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def export_results(self, results: List[Dict[str, Any]], base_filename: str = "test_results") -> None:
        """Export results to CSV file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.csv"
        
        if not results:
            print("No results to export.")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_id', 'model', 'prompt', 'query', 'response', 'analysis_text',
                'response_time', 'success', 'error', 'timestamp'
            ])
            writer.writeheader()
            for result in results:
                response_text = str(result.get('response', ''))
                analysis_text = self._extract_analysis_text(response_text)
                
                writer.writerow({
                    'test_id': result.get('test_id', ''),
                    'model': result.get('model', ''),
                    'prompt': result.get('prompt', ''),
                    'query': result.get('query', ''),
                    'response': response_text,
                    'analysis_text': analysis_text,
                    'response_time': result.get('response_time', 0),
                    'success': result.get('success', False),
                    'error': result.get('error', ''),
                    'timestamp': result.get('timestamp', '')
                })
        
        print(f"📁 Results exported to {filename}")

    def list_models_by_provider(self, filter_string: str = None):
        """List all models grouped by provider in a human-readable format
        
        Args:
            filter_string: Optional string to filter models (case-insensitive)
        """
        provider_groups = {}
        filtered_models = {}
        
        # Filter models if filter_string is provided
        if filter_string:
            filter_string = filter_string.lower()
            for model_name, config in self.models.items():
                if filter_string in model_name.lower() or filter_string in config["model_id"].lower():
                    filtered_models[model_name] = config
        else:
            filtered_models = self.models
        
        for model_name, config in filtered_models.items():
            model_id = config["model_id"]
            if model_id.startswith("anthropic") or "anthropic" in model_id:
                provider = "Anthropic"
            elif model_id.startswith("amazon") or "amazon" in model_id:
                provider = "Amazon"
            elif model_id.startswith("meta") or "meta" in model_id:
                provider = "Meta"
            elif model_id.startswith("mistral") or "mistral" in model_id:
                provider = "Mistral AI"
            elif model_id.startswith("cohere"):
                provider = "Cohere"
            elif model_id.startswith("deepseek") or "deepseek" in model_id:
                provider = "DeepSeek"
            elif model_id.startswith("openai") or "openai" in model_id:
                provider = "OpenAI"
            elif model_id.startswith("ai21"):
                provider = "AI21"
            elif "twelvelabs" in model_id:
                provider = "TwelveLabs"
            else:
                provider = "Other"
            
            if provider not in provider_groups:
                provider_groups[provider] = []
            
            provider_groups[provider].append({
                "name": model_name,
                "model_id": model_id[0:28],
                "region": config["region_name"],
                "type": config["inference_type"],
                "tooling": config["tooling_enabled"]
            })
        
        filter_text = f" (filtered by '{filter_string}')" if filter_string else ""
        print(f"\n📋 AVAILABLE MODELS ({len(filtered_models)} total{filter_text})")
        print("=" * 140)
        
        for provider in sorted(provider_groups.keys()):
            models = sorted(provider_groups[provider], key=lambda x: x["name"])
            print(f"\n🏢 {provider} ({len(models)} models)")
            print("-" * 90)
            print(f"  {'TYPE':<4}| {'MODEL_NAME':<25} | {'ENDPOINT':<33} | {'REGION':<13} | {'TOOL_SUPPORT'}")
            print("-" * 90)
            
            for model in models:
                region_badge = "🌍" if model["type"] == "INFERENCE_PROFILE" else "📍"
                tooling_badge = "🔧" if model["tooling"] else "❌"
                print(f"  {region_badge:<4} {model['name']:<25} | {model['model_id']:<32} | {model['region']:<12} | {tooling_badge}")


    def run_evaluation(self, 
                      models: List[str], 
                      system_prompts: List[str], 
                      prompts_dict: Dict[str, str], 
                      tool: List = None,
                      test_cases_path: str = None,
                      langfuse_public_key: str = None,
                      langfuse_secret_key: str = None, 
                      langfuse_api_url: str = None,
                      save_to_csv: bool = True) -> List[Dict[str, Any]]:
        """
        Run evaluation using test cases from YAML file with LLM-as-judge evaluation.
        
        Args:
            models: List of model IDs to test
            system_prompts: List of prompt versions to test
            prompts_dict: Dictionary containing system prompts
            tool: List of tools to use
            test_cases_path: Path to YAML file containing test cases
            langfuse_public_key: Optional Langfuse public key for tracing
            langfuse_secret_key: Optional Langfuse secret key for tracing
            langfuse_api_url: Optional Langfuse API URL for tracing
            save_to_csv: Whether to save results to CSV (default: True)
            
        Returns: List of evaluation results
        """
        import yaml
        import re
        
        if not test_cases_path:
            raise ValueError("test_cases_path is required")
        
        # Load test cases from YAML
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            test_cases = yaml.safe_load(f)
        
        # Initialize Langfuse if credentials provided
        langfuse_client = None
        main_trace = None
        if langfuse_public_key and langfuse_secret_key and langfuse_api_url:
            try:
                from langfuse import Langfuse
                langfuse_client = Langfuse(
                    public_key=langfuse_public_key,
                    secret_key=langfuse_secret_key,
                    host=langfuse_api_url
                )
                
                # Create main trace for the entire evaluation (root span)
                main_trace = langfuse_client.start_span(
                    name="Agent Eval - " + (models[0] if models else 'Multiple Models'),
                    input={
                        "config_file": test_cases_path,
                        "evaluation_metadata": {
                            "models": models,
                            "system_prompts": system_prompts,
                            "test_cases": list(test_cases.keys()),
                            "total_combinations": len(models) * len(system_prompts) * len(test_cases),
                            "evaluation_timestamp": datetime.now().isoformat()
                        }
                    },
                    metadata={
                        "evaluation_framework": "Strands Agents Test Evaluator",
                        "version": "1.0",
                        "langfuse_version": "v3"
                    }
                )
                
                print("✅ Langfuse tracing enabled")
            except ImportError:
                print("⚠️ Langfuse not available, continuing without tracing")
        
        results = []
        total_combinations = len(models) * len(system_prompts) * len(test_cases)
        current_combination = 0
        
        print(f"\n🧪 Starting Test Case Evaluation")
        print(f"📊 Total combinations: {total_combinations}")
        print(f"🤖 Models: {models}")
        print(f"📝 Prompts: {system_prompts}")
        print(f"📋 Test Cases: {len(test_cases)} test case(s)")
        print("=" * 80)
        
        for model_name in models:
            for prompt_name in system_prompts:
                for test_name, test_data in test_cases.items():
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] Evaluating: {model_name} | {prompt_name} | {test_name}")
                    print("-" * 60)
                    
                    # Create evaluator agent for this combination
                    evaluator_agent = self._create_agent(model_name, prompt_name, prompts_dict, tool)
                    
                    # Create judge agent (using same model for simplicity)
                    judge_agent = self._create_judge_agent(model_name)
                    
                    # Run evaluation for this test case
                    eval_result = self._evaluate_test_case(
                        test_name, test_data, evaluator_agent, judge_agent, 
                        model_name, prompt_name, langfuse_client, main_trace
                    )
                    
                    results.append(eval_result)
                    
                    # Brief status
                    status = "✅ PASSED" if eval_result['passed'] else "❌ FAILED"
                    print(f"{status} | Score: {eval_result['score']:.2f}")
        
        print(f"\n🎉 Evaluation Completed! {len(results)} results generated.")
        
        # Finalize Langfuse main trace if available
        if main_trace and langfuse_client:
            try:
                # Calculate overall metrics
                total_tests = len(results)
                passed_tests = sum(1 for r in results if r['passed'])
                overall_score = float(passed_tests) / float(total_tests) if total_tests > 0 else 0.0
                overall_result = "PASS" if passed_tests == total_tests else "FAILED"
                
                # Update main trace with final results
                main_trace.update(
                    input={
                        "config_file": test_cases_path,
                        "evaluation_metadata": {
                            "models": models,
                            "system_prompts": system_prompts,
                            "test_cases": list(test_cases.keys()),
                            "total_combinations": len(models) * len(system_prompts) * len(test_cases),
                            "evaluation_timestamp": datetime.now().isoformat()
                        }
                    },
                    output={
                        "overall_result": overall_result,
                        "overall_results": {
                            "total_tests": total_tests,
                            "passed_tests": passed_tests,
                            "pass_rate": f"{(overall_score * 100):.1f}%",
                            "overall_passed": passed_tests == total_tests
                        }
                    }
                )
                
                # Update main trace with final results
                main_trace.update_trace(
                    tags=[f"Agent-Eval-{models[0] if models else 'Multiple-Models'}"],
                    input={
                        "Eval File": test_cases_path
                    },
                    output={
                        "Test Passed": str(passed_tests == total_tests).upper()
                    }
                )
                
                # End main trace
                main_trace.end()
                
                # Add overall score
                langfuse_client.create_score(
                    trace_id=main_trace.trace_id,
                    observation_id=main_trace.id,
                    name="overall_evaluation",
                    value=overall_score,
                    data_type="NUMERIC",
                    comment=f"Overall evaluation: {passed_tests}/{total_tests} tests passed ({(overall_score * 100):.1f}%)"
                )
                
                # Flush data
                langfuse_client.flush()
                print(f"✅ Langfuse trace finalized: {main_trace.trace_id}")
                
            except Exception as e:
                print(f"⚠️ Error finalizing Langfuse trace: {str(e)}")
        
        # Display summary
        self._display_evaluation_summary(results)
        
        # Save to CSV if requested
        if save_to_csv:
            self._export_evaluation_results(results)
        
        return results

    def _create_judge_agent(self, model_name: str) -> Agent:
        """Create a judge agent for evaluation"""
        model_config = self.models[model_name]
        
        judge_system_prompt = """You are an expert quality assurance engineer evaluating an agent's response to a user question.

Your job is to analyze the user question, agent response, and expected result to determine if the agent's response meets the expected criteria.

You MUST classify the response into one of these categories:
- PASSED: The agent's response meets or exceeds the expected result criteria
- FAILED: The agent's response does not meet the expected result criteria

CRITICAL: You MUST format your response exactly as follows:

<analysis>
[Your one paragraph detailed analysis of whether the response meets the criteria]
</analysis>

<category>PASSED</category>

OR

<category>FAILED</category>

Do not include any other text outside these XML tags."""
        
        model = BedrockModel(
            model_id=model_config["model_id"],
            temperature=0.1,  # Lower temperature for consistent evaluation
            region_name=model_config["region_name"]
        )
        
        return Agent(model=model, system_prompt=judge_system_prompt)

    def _evaluate_test_case(self, test_name: str, test_data: Dict, 
                           evaluator_agent: Agent, judge_agent: Agent,
                           model_name: str, prompt_name: str,
                           langfuse_client=None, main_trace=None) -> Dict[str, Any]:
        """Evaluate a single test case with multi-turn conversation"""
        """Evaluate a single test case with multi-turn conversation"""
        
        # Extract questions and expected results
        questions = []
        expected_results = []
        
        for question_key in sorted(test_data.keys()):
            if question_key.startswith('question_'):
                questions.append(test_data[question_key]['question'])
                expected_results.append(test_data[question_key]['expected_results'])
        
        conversation_history = []
        question_results = []
        
        print(f"📝 Test Case: {test_name}")
        
        # Process each question as a separate turn
        for i, (question, expected_result) in enumerate(zip(questions, expected_results)):
            print(f"\n  Turn {i + 1}: {question[:60]}{'...' if len(question) > 60 else ''}")
            
            try:
                # Get agent response
                start_time = time.time()
                response = evaluator_agent(question)
                end_time = time.time()
                
                # Extract response text
                if hasattr(response, 'message') and 'content' in response.message:
                    agent_response = response.message['content']
                else:
                    agent_response = str(response)
                
                conversation_history.append(("USER", question))
                conversation_history.append(("AGENT", agent_response))
                
                # Evaluate this question-answer pair
                print(f"\n{'='*50}")
                print("AGENT EVALUATION RESULTS")
                print(f"{'='*50}")
                eval_category, reasoning = self._judge_response(
                    judge_agent, expected_result, question, agent_response
                )
                print(f"{'='*50}")
                
                question_passed = eval_category.strip().upper() == "PASSED"
                
                question_result = {
                    "question_number": i + 1,
                    "question": question,
                    "expected_result": expected_result,
                    "agent_response": agent_response,
                    "passed": question_passed,
                    "reasoning": reasoning,
                    "response_time": end_time - start_time
                }
                question_results.append(question_result)
                
                status = "✅" if question_passed else "❌"
                print(f"{status} Question {i + 1}")
                
            except Exception as e:
                print(f"    ❌ Error in question {i + 1}: {str(e)}")
                question_result = {
                    "question_number": i + 1,
                    "question": question,
                    "expected_result": expected_result,
                    "agent_response": f"Error: {str(e)}",
                    "passed": False,
                    "reasoning": f"Error occurred: {str(e)}",
                    "response_time": 0
                }
                question_results.append(question_result)
        
        # Calculate overall test result
        passed_questions = sum(1 for r in question_results if r['passed'])
        total_questions = len(question_results)
        overall_passed = passed_questions == total_questions
        score = passed_questions / total_questions if total_questions > 0 else 0.0
        
        # Create evaluation result
        eval_result = {
            "test_id": f"{model_name}_{prompt_name}_{test_name}",
            "model": model_name,
            "prompt": prompt_name,
            "test_name": test_name,
            "passed": overall_passed,
            "score": score,
            "passed_questions": passed_questions,
            "total_questions": total_questions,
            "conversation": conversation_history,
            "question_results": question_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add Langfuse tracing if available - create spans for each question
        if langfuse_client and main_trace:
            try:
                
                # Create spans for each question directly as children of main trace (like reference)
                for i, question_result in enumerate(question_results):
                    question_span = main_trace.start_span(
                        name=f"Question: {test_name} - Q{i+1}",
                        input={
                            "test_name": test_name,
                            "question_id": f"{test_name}_q{i+1}",
                            "question": question_result['question'],
                            "expected_result": question_result['expected_result']
                        },
                        output={
                            "agent_response": question_result['agent_response'],
                            "question_passed": question_result['passed'],
                            "reasoning": question_result['reasoning']
                        },
                        metadata={
                            "question_number": i + 1,
                            "test_passed": overall_passed,
                            "evaluation_category": "A" if question_result['passed'] else "B",
                            "test_name": test_name
                        },
                        level="DEFAULT"
                    )
                    
                    # Update and end the question span
                    question_span.update(
                        output={
                            "agent_response": question_result['agent_response'],
                            "question_passed": question_result['passed'],
                            "reasoning": question_result['reasoning']
                        }
                    )
                    question_span.end()
                    
                    # Add score to the question span
                    langfuse_client.create_score(
                        trace_id=main_trace.trace_id,
                        observation_id=question_span.id,
                        name="question_evaluation",
                        value=1.0 if question_result['passed'] else 0.0,
                        data_type="NUMERIC",
                        comment=question_result['reasoning'] or "No reasoning provided"
                    )

                
                print(f"  ✅ Created {len(question_results)} question spans for test: {test_name}")
                
            except Exception as e:
                print(f"⚠️ Langfuse tracing error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return eval_result

    def _judge_response(self, judge_agent: Agent, expected_result: str, 
                       question: str, agent_response: str) -> Tuple[str, str]:
        """Use judge agent to evaluate response"""
        
        prompt = f"""Here is the evaluation scenario:

<question>
{question}
</question>

<agent_response>
{agent_response}
</agent_response>

<expected_result>
{expected_result}
</expected_result>

Evaluate whether the agent's response meets the expected result criteria or not."""
        
        try:
            response = judge_agent(prompt)
            
            # Handle different response formats
            if hasattr(response, 'message') and 'content' in response.message:
                completion = response.message['content']
            else:
                completion = str(response)
            
            # Handle list responses (Bedrock sometimes returns lists)
            if isinstance(completion, list):
                if len(completion) > 0 and isinstance(completion[0], dict) and 'text' in completion[0]:
                    completion = completion[0]['text']
                else:
                    completion = str(completion)
            
            # Extract category and reasoning from XML tags
            category, reasoning = self._extract_xml_content(completion)
            
            return category, reasoning
            
        except Exception as e:
            return "FAILED", f"Evaluation error: {str(e)}"

    def _extract_xml_content(self, text: str) -> Tuple[str, str]:
        """Extract content from XML tags"""
        import re
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        
        # Extract analysis text from <analysis> tags
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', text, re.DOTALL | re.IGNORECASE)
        analysis_text = analysis_match.group(1).strip() if analysis_match else "Analysis complete"
        
        # Simple, bulletproof extraction using string matching
        if '<category>PASSED</category>' in text:
            return "PASSED", analysis_text
        elif '<category>FAILED</category>' in text:
            return "FAILED", analysis_text
        elif '<category>A</category>' in text:
            return "PASSED", analysis_text  
        elif '<category>B</category>' in text:
            return "FAILED", analysis_text
        else:
            # Fallback for any edge cases
            if 'PASSED' in text.upper():
                return "PASSED", analysis_text
            else:
                return "FAILED", analysis_text

    def _display_evaluation_summary(self, results: List[Dict[str, Any]]) -> None:
        """Display evaluation results summary"""
        if not results:
            print("No evaluation results to display.")
            return
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['passed'])
        avg_score = sum(r['score'] for r in results) / total_tests
        
        print(f"\n📊 EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Test Cases: {total_tests}")
        print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"❌ Failed: {total_tests - passed_tests} ({(total_tests - passed_tests)/total_tests*100:.1f}%)")
        print(f"📈 Average Score: {avg_score:.2f}")
        
        # Group by model and prompt
        by_model = defaultdict(list)
        by_prompt = defaultdict(list)
        
        for result in results:
            by_model[result['model']].append(result)
            by_prompt[result['prompt']].append(result)
        
        # Model performance
        print(f"\n🤖 MODEL PERFORMANCE")
        print(f"{'='*50}")
        for model, model_results in by_model.items():
            model_avg_score = sum(r['score'] for r in model_results) / len(model_results)
            model_passed = sum(1 for r in model_results if r['passed'])
            print(f"{model:20} | Score: {model_avg_score:5.2f} | Passed: {model_passed}/{len(model_results)}")
        
        # Prompt performance
        print(f"\n📝 PROMPT PERFORMANCE")
        print(f"{'='*50}")
        for prompt, prompt_results in by_prompt.items():
            prompt_avg_score = sum(r['score'] for r in prompt_results) / len(prompt_results)
            prompt_passed = sum(1 for r in prompt_results if r['passed'])
            print(f"{prompt:20} | Score: {prompt_avg_score:5.2f} | Passed: {prompt_passed}/{len(prompt_results)}")

    def _export_evaluation_results(self, results: List[Dict[str, Any]], 
                                  base_filename: str = "evaluation_results") -> None:
        """Export evaluation results to CSV"""
        import os
        
        # Create evaluation_results directory if it doesn't exist
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(eval_dir, f"{base_filename}_{timestamp}.csv")
        
        if not results:
            print("No evaluation results to export.")
            return
        
        # Flatten results to include individual question data with analysis
        flattened_results = []
        for result in results:
            question_results = result.get('question_results', [])
            for q_result in question_results:
                flattened_results.append({
                    'test_id': result.get('test_id', ''),
                    'model': result.get('model', ''),
                    'prompt': result.get('prompt', ''),
                    'test_name': result.get('test_name', ''),
                    'question_number': q_result.get('question_number', ''),
                    'question': q_result.get('question', ''),
                    'agent_response': q_result.get('agent_response', ''),
                    'analysis_text': q_result.get('reasoning', ''),
                    'passed': q_result.get('passed', False),
                    'response_time': q_result.get('response_time', 0),
                    'timestamp': result.get('timestamp', '')
                })
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_id', 'model', 'prompt', 'test_name', 'question_number', 'question',
                'agent_response', 'analysis_text', 'passed', 'response_time', 'timestamp'
            ])
            writer.writeheader()
            for row in flattened_results:
                writer.writerow(row)
        
        print(f"📁 Evaluation results exported to {filename}")
