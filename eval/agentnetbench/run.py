#!/usr/bin/env python3
import os
import json
import argparse
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict

from agent.base_agent import BaseAgent
from agent.aguvis import Aguvis
from agent.qwen25vl import Qwen25VL
from agent.opencua import OpenCUA
from eval import ActionEvaluator

"""
Benchmark evaluation runner supporting Qwen2.5-VL and Aguvis agents.
"""

class TrajectoryEvaluator:
    """Class to handle trajectory evaluation using different agents."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the evaluator with command line arguments."""
        self.args = args
        self.output_dir = self._setup_output_dir()

    def _setup_output_dir(self) -> Path:
        """Create and return the output directory path."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self.args.model.replace('/', '_').replace('\\', '_')  # Sanitize model name for filesystem
        output_dir = Path(self.args.output) / f"eval_{timestamp}_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Removed _create_agent: async path below creates the agent as needed

    def load_trajectories(self) -> List[Dict[str, Any]]:
        """Load all trajectory data from the data directory."""
        data_dir = Path(self.args.data)
        trajectories = []
        
        for file in data_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    trajectory = json.load(f)
                    trajectories.append(trajectory)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        return trajectories

    def save_result(self, trajectory: Dict[str, Any], results: List[Dict[str, Any]]):
        """Save evaluation results for a trajectory."""
        output_file = self.output_dir / f"{trajectory['task_id']}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    def save_metrics(self, metrics: Dict[str, Any]):
        """Save evaluation metrics."""
        metric_file = self.output_dir / "metric.json"
        with open(metric_file, "w") as f:
            json.dump(metrics, f, indent=4)

    def save_hyperparams(self):
        """Save hyperparameters and configuration."""
        params = {
            "base_url": self.args.base_url,
            "data_path": str(self.args.data),
            "model": self.args.model,
            "image_dir": str(self.args.image_dir)
        }
        
        param_file = self.output_dir / "hyperparams.json"
        with open(param_file, "w") as f:
            json.dump(params, f, indent=4)

    async def evaluate_trajectory_parallel(self, trajectories: List[Dict[str, Any]], num_cores: int = 4) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Evaluate trajectories in parallel using asyncio.
        
        This implementation processes all trajectories as fast as possible and saves results immediately when each trajectory completes.
        
        Args:
            trajectories: List of trajectories to evaluate
            num_cores: Number of parallel clients to use
            
        Returns:
            List of tuples containing (task_id, results) for each trajectory
        """
        # Create a client first since some agent types might not support the create method
        # IMPORTANT: Always use AsyncOpenAI for async operations
        from openai import AsyncOpenAI
        
        # Explicitly print the credentials being used
        print(f"Creating AsyncOpenAI client with base_url: {self.args.base_url}")
        print(f"Using API key: {self.args.api_key[:4]}...{self.args.api_key[-4:] if len(self.args.api_key) > 8 else ''}")
        
        # Create the client with explicit credentials
        client = AsyncOpenAI(
            base_url=self.args.base_url, 
            api_key=self.args.api_key
        )
        
        # Store the API credentials in BaseAgent class for use in pool initialization
        from agent.base_agent import BaseAgent
        BaseAgent._base_url = self.args.base_url
        BaseAgent._api_key = self.args.api_key
        
        # Get the appropriate agent type based on model name
        model_name = self.args.model.lower()
        
        # Check for Qwen models - support various naming formats
        if any(qwen_name in model_name for qwen_name in ["qwen2.5-vl", "qwen-vl", "qwen25vl", "qwen2.5vl"]):
            print(f"Using Qwen25VL agent for model: {self.args.model}")
            agent_class = Qwen25VL
            agent = agent_class(self.args.model, client)
            
        # Aguvis models
        elif "aguvis" in model_name:
            print(f"Using Aguvis agent for model: {self.args.model}")
            agent_class = Aguvis
            agent = agent_class(self.args.model, client)

        # OpenCUA models
        elif "opencua" in model_name:
            print(f"Using OpenCUA agent for model: {self.args.model}")
            agent_class = OpenCUA
            agent = agent_class(
                self.args.model,
                client,
                l_number=self.args.opencua_l_number,
                history=self.args.opencua_history,
                image=self.args.opencua_image,
                max_history_length=self.args.opencua_max_history_length,
                max_detail_length=self.args.opencua_max_detail_length,
            )

        # Unsupported
        else:
            valid_models = ["aguvis", "qwen2.5-vl", "qwen25vl", "opencua"]
            raise ValueError(f"Unsupported model type: {self.args.model}. Valid types include: {', '.join(valid_models)}")
        
        evaluator = ActionEvaluator()
        
        async def process_trajectory(trajectory: Dict[str, Any], worker_id: int) -> Tuple[str, List[Dict[str, Any]]]:
            """Process a single trajectory asynchronously."""
            task_id = trajectory.get('task_id', 'unknown')
            
            try:
                # Process trajectory and get results - use the same agent for all trajectories
                agent_results = await agent.test_traj_async(trajectory, self.args.image_dir)
                results = []
                
                for step_result in agent_results:
                    step_data = trajectory["steps"][step_result["step_idx"]]
                    
                    # Create base result
                    result = {
                        "task_id": trajectory["task_id"],
                        "step_num": step_result["step_idx"],
                        "ground_truth_actions": step_data.get("ground_truth_actions", []),
                        "milestone": step_data.get("milestone", False),
                        "raw_response": step_result["raw_response"],
                        "parsed_action": step_result.get("parsed_action"),
                        "predicted_actions": step_result.get("actions", [])
                    }
                    
                    # Handle parsing errors
                    if step_result.get("parsing_error", False):
                        result.update({
                            "used_actions": [],
                            "alternative_matched": False,
                            "parsing_error": True,
                            "error_type": step_result.get("error_type", "unknown"),
                            "error_message": step_result.get("error_message", ""),
                            "evaluation": {"total": 0, "actions": {}}
                        })
                        results.append(result)
                        continue
                    
                    # Evaluate actions if present
                    if step_result.get("actions"):
                        gt_eval_item = {
                            "ground_truth_actions": step_data.get("ground_truth_actions", []),
                            "predicted_actions": step_result["actions"]
                        }
                        gt_eval_scores = evaluator.evaluate_action(gt_eval_item)
                        best_eval_scores = gt_eval_scores
                        used_actions = step_data.get("ground_truth_actions", [])
                        
                        # Check alternative options
                        if gt_eval_scores["total"] < 1.0 and "alternative_options" in step_data:
                            for alt_option in step_data.get("alternative_options", []):
                                alt_eval_item = {
                                    "ground_truth_actions": alt_option,
                                    "predicted_actions": step_result["actions"]
                                }
                                alt_eval_scores = evaluator.evaluate_action(alt_eval_item)
                                if alt_eval_scores["total"] > best_eval_scores["total"]:
                                    best_eval_scores = alt_eval_scores
                                    used_actions = alt_option
                        
                        result.update({
                            "used_actions": used_actions,
                            "alternative_matched": used_actions != step_data.get("ground_truth_actions", []),
                            "evaluation": best_eval_scores
                        })
                    else:
                        result.update({
                            "used_actions": [],
                            "alternative_matched": False,
                            "evaluation": {"total": 0, "actions": {}}
                        })
                    
                    results.append(result)
                
                # Save results immediately
                self.save_result(trajectory, results)
                return task_id, results
                
            except Exception as e:
                print(f"Failed to evaluate trajectory {task_id}: {e}")
                # Create error result and save immediately
                error_result = [{
                    "task_id": task_id,
                    "step_num": 0,
                    "raw_response": f"Evaluation failed: {str(e)}",
                    "ground_truth_actions": [],
                    "milestone": False,
                    "parsed_action": None,
                    "predicted_actions": [],
                    "used_actions": [],
                    "alternative_matched": False,
                    "parsing_error": True,
                    "error_type": "evaluation_error",
                    "error_message": str(e),
                    "evaluation": {"total": 0, "actions": {}}
                }]
                self.save_result(trajectory, error_result)
                return task_id, error_result
        
        # Create all tasks at once
        tasks = [
            process_trajectory(trajectory, i) 
            for i, trajectory in enumerate(trajectories)
        ]
        
        # Process all trajectories concurrently and collect results
        all_results = []
        completed = 0
        total = len(trajectories)
        
        # Process as they complete
        for task in asyncio.as_completed(tasks):
            task_id, results = await task
            all_results.append((task_id, results))
            completed += 1
            print(f"Completed {completed}/{total} trajectories ({(completed/total*100):.1f}%)")
        
        return all_results

    async def run_evaluation_async(self):
        """Run the complete evaluation process asynchronously."""
        print("\n=== Starting Evaluation ===")
        print(f"Model: {self.args.model}")
        print(f"Data directory: {self.args.data}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using {self.args.num_cores} cores for parallel processing")
        
        # Load trajectories
        trajectories = self.load_trajectories()
        print(f"Loaded {len(trajectories)} trajectories")
        
        # Evaluate trajectories in parallel
        results = await self.evaluate_trajectory_parallel(trajectories, self.args.num_cores)
        
        # Save results
        completed = 0
        failed = 0
        for task_id, trajectory_results in results:
            # Find the matching trajectory
            trajectory = next((t for t in trajectories if t["task_id"] == task_id), None)
            
            if trajectory is None:
                print(f"WARNING: Could not find trajectory for task_id {task_id}")
                # Create a dummy trajectory if needed
                trajectory = {"task_id": task_id}
                failed += 1
            
            # Even if results are empty, save a placeholder result to indicate the run was attempted
            if not trajectory_results:
                trajectory_results = [{
                    "task_id": task_id,
                    "step_num": 0,
                    "raw_response": "No valid results produced for this trajectory",
                    "ground_truth_actions": [],
                    "alternative_options": [],
                    "milestone": False,
                    "parsed_action": None,
                    "predicted_actions": [],
                    "used_actions": [],
                    "alternative_matched": False,
                    "parsing_error": True,
                    "error_type": "no_results",
                    "evaluation": {"total": 0, "actions": {}}
                }]
                failed += 1
            else:
                completed += 1
            
            # Save the results to file
            self.save_result(trajectory, trajectory_results)
        
        # Calculate and save metrics
        metrics = self.calculate_metrics()
        self.save_metrics(metrics)
        self.save_hyperparams()
        
        # Print summary
        print("\n=== Evaluation Complete ===")
        print(f"Completed: {completed}/{len(trajectories)} trajectories")
        print(f"Failed: {failed} trajectories")
        print(f"Results saved to: {self.output_dir}")
        print("\nMetrics Summary:")
        print(f"Average Score: {metrics['average_score']:.3f}")
        print(f"Alternative Matches: {metrics['alternative_matches']} steps ({metrics['alternative_match_percentage']:.1f}%)")
        print("\nAction Type Scores:")
        # Sort action types by their count in the metrics
        sorted_actions = sorted(metrics['action_type_scores'].items(), 
                              key=lambda x: len([a for t in trajectories for a in t['steps'] if a.get('action_type') == x[0]]),
                              reverse=True)
        for action_type, score in sorted_actions:
            print(f"  {action_type}: {score:.3f}")

    def run_evaluation(self):
        """Run the complete evaluation process."""
        asyncio.run(self.run_evaluation_async())

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        total_steps = 0
        total_score = 0.0
        action_type_scores = defaultdict(list)
        alt_matched_count = 0
        
        # Process each result file
        for result_file in self.output_dir.glob("*.json"):
            if result_file.name in ["metric.json", "hyperparams.json"]:
                continue
                
            with open(result_file, "r") as f:
                results = json.load(f)
                
            for result in results:
                if "evaluation" not in result:
                    continue
                    
                total_steps += 1
                total_score += result["evaluation"]["total"]
                
                # Track when alternative actions were used for matching
                if result.get("alternative_matched", False):
                    alt_matched_count += 1
                
                # Collect scores by action type
                for action_type, score in result["evaluation"]["actions"].items():
                    action_type_scores[action_type].append(score)
                    
                # No milestone-specific metrics currently
        
        # Calculate metrics
        metrics = {
            "total_trajectories": len(list(self.output_dir.glob("*.json"))) - 2,
            "total_steps": total_steps,
            "average_score": total_score / total_steps if total_steps > 0 else 0.0,
            "alternative_matches": alt_matched_count,
            "alternative_match_percentage": (alt_matched_count / total_steps * 100) if total_steps > 0 else 0.0,
            "action_type_scores": {
                action_type: sum(scores) / len(scores)
                for action_type, scores in action_type_scores.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectories using different agents")
    parser.add_argument("--data", type=str, default="./test_data",
                      help="Directory containing trajectory JSON files")
    parser.add_argument("--image_dir", type=str, default="./test_data/images",
                      help="Directory containing trajectory images")
    parser.add_argument("--output", type=str, default="./output",
                      help="Output directory for evaluation results")
    parser.add_argument("--model", type=str, default="qwen25vl",
                      help="Model to use for evaluation (aguvis, qwen25vl, opencua)")
    parser.add_argument("--base_url", type=str, default=None,
                      help="Base URL of the hosted model service")
    parser.add_argument("--api_key", type=str, default=None,
                      help="API key for the hosted model service")
    parser.add_argument("--num_cores", type=int, default=10,
                      help="Number of CPU cores to use for parallel processing")

    # OpenCUA-specific options
    parser.add_argument("--opencua-l-number", dest="opencua_l_number", type=str, default="l2",
                      choices=["l1", "l2", "l3", "l1_short", "l2_short", "l3_short"],
                      help="OpenCUA prompt style level")
    parser.add_argument("--opencua-history", dest="opencua_history", type=str, default="thought",
                      choices=["action", "thought", "observation"],
                      help="OpenCUA history mode to include in prompts")
    parser.add_argument("--opencua-image", dest="opencua_image", type=str, default="image_3",
                      choices=["image_1", "image_3", "image_5"],
                      help="How many previous images to include (1/3/5)")
    parser.add_argument("--opencua-max-history-length", dest="opencua_max_history_length", type=int, default=10,
                      help="Maximum number of previous steps to include as text history")
    parser.add_argument("--opencua-max-detail-length", dest="opencua_max_detail_length", type=int, default=0,
                      help="For last N history steps, include full detailed responses (observation/thought/action/code)")
    
    args = parser.parse_args()
    
    evaluator = TrajectoryEvaluator(args)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
