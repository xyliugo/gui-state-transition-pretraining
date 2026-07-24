# AgentNetBench Evaluation

Evaluates GUI agent performance on the AgentNetBench desktop benchmark.

## Metrics

- **Coordinate Success Rate**: Whether predicted click coordinates fall within the target element bounding box
- **Content Success Rate**: Text similarity between predicted and ground truth text actions
- **Function Success Rate**: Whether the predicted action type matches ground truth
- **Average Success Rate**: Mean of all metrics

## Usage

```bash
python eval/agentnetbench/run.py \
  --data ${DATA_ROOT}/agentnetbench/test_data \
  --image_dir ${DATA_ROOT}/agentnetbench/test_data/images \
  --output results/eval_agentnetbench \
  --model your_model_name \
  --base_url http://localhost:8000/v1 \
  --api_key YOUR_KEY
```

## Output

Results are saved to `output/eval_YYYYMMDD_HHMMSS_<model>/`:
- `metric.json`: Aggregated metrics
- `<task_id>.json`: Per-trajectory detailed results
