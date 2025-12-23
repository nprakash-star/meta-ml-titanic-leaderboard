"""
Meta-ML Green Agent (Evaluator)
Evaluates ML agents on performance and research quality.
"""
import pandas as pd
import io
import requests
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, DataPart, TextPart
from a2a.utils import new_agent_text_message
from messenger import Messenger
from sklearn.metrics import accuracy_score, f1_score


class Agent:
    def __init__(self):
        self.messenger = Messenger()
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main evaluation logic."""
        
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message("Green Agent: Starting evaluation...")
        )
        
        # Extract metadata
        metadata = message.metadata or {}
        purple_agent_url = metadata.get("agent_url")
        
        if not purple_agent_url:
            await updater.failed(new_agent_text_message("No agent_url provided in metadata"))
            return
        
        # Get dataset URLs from config
        train_url = metadata.get("train_data_url")
        test_url = metadata.get("test_data_url")
        labels_url = metadata.get("test_labels_url")
        target_col = metadata.get("target_column")
        task_desc = metadata.get("task_description", "Train ML model")
        
        if not train_url or not test_url or not labels_url:
            await updater.failed(new_agent_text_message(
                "Missing dataset URLs. Need train_data_url, test_data_url, test_labels_url in metadata"
            ))
            return
        
        # Step 1: Download datasets
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(f"Downloading datasets from GitHub...")
        )
        
        try:
            train_csv = requests.get(train_url).text
            test_csv = requests.get(test_url).text
            labels_csv = requests.get(labels_url).text
            
            # Parse labels for later evaluation
            labels_df = pd.read_csv(io.StringIO(labels_csv))
            if target_col and target_col in labels_df.columns:
                y_true = labels_df[target_col]
            else:
                # Try common label column names
                for col in ['label', 'target', 'y', target_col]:
                    if col and col in labels_df.columns:
                        y_true = labels_df[col]
                        break
                else:
                    y_true = labels_df.iloc[:, 0]
            
            y_true = [str(x) for x in y_true]
            
        except Exception as e:
            await updater.failed(new_agent_text_message(f"Failed to download datasets: {str(e)}"))
            return
        
        # Step 2: Call Purple Agent
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(f"Calling solver agent at {purple_agent_url}...")
        )
        
        try:
            # Create task message for purple agent
            task_message = Message(
                kind="message",
                message_id="eval_task",
                role="user",
                parts=[
                    Part(root=TextPart(kind="text", text=f"Train a model: {task_desc}")),
                    Part(root=DataPart(kind="data", data={"filename": "train.csv", "content": train_csv})),
                    Part(root=DataPart(kind="data", data={"filename": "test.csv", "content": test_csv}))
                ],
                metadata={
                    "task_description": task_desc,
                    "target_column": target_col
                }
            )
            
            # Send to purple agent directly using HTTP client
            import httpx
            from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
            
            async with httpx.AsyncClient(timeout=600) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=purple_agent_url)
                agent_card = await resolver.get_agent_card()
                config = ClientConfig(httpx_client=httpx_client, streaming=False)
                factory = ClientFactory(config)
                client = factory.create(agent_card)
                
                last_event = None
                async for event in client.send_message(task_message):
                    last_event = event
                
                # Extract response
                if isinstance(last_event, tuple):
                    response, _ = last_event
                else:
                    response = last_event
            
            # Extract predictions and research from response
            predictions_text = None
            research_text = ""
            
            if hasattr(response, 'artifacts'):
                for artifact in response.artifacts:
                    if artifact.name == "predictions":
                        predictions_text = self._extract_text(artifact)
                    elif artifact.name == "research_report":
                        research_text = self._extract_text(artifact)
            
            if not predictions_text:
                await updater.failed(new_agent_text_message("Purple agent did not return predictions"))
                return
            
        except Exception as e:
            import traceback
            await updater.failed(new_agent_text_message(
                f"Failed to communicate with purple agent: {str(e)}\n{traceback.format_exc()}"
            ))
            return
        
        # Step 3: Evaluate predictions
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message("Evaluating predictions...")
        )
        
        try:
            # Parse predictions
            pred_lines = [line.strip() for line in predictions_text.split('\n') if line.strip()]
            # Skip header if present
            if pred_lines and pred_lines[0].lower() == 'prediction':
                pred_lines = pred_lines[1:]
            
            y_pred = [str(x) for x in pred_lines]
            
            # Align lengths
            min_len = min(len(y_true), len(y_pred))
            y_true_aligned = y_true[:min_len]
            y_pred_aligned = y_pred[:min_len]
            
            # Compute metrics
            accuracy = float(accuracy_score(y_true_aligned, y_pred_aligned))
            f1 = float(f1_score(y_true_aligned, y_pred_aligned, average="weighted"))
            
            # Simple research scoring
            research_score = self._score_research(research_text)
            
            # Compile results
            eval_results = {
                "performance": {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "num_samples": min_len
                },
                "research": {
                    "score": research_score,
                    "length": len(research_text)
                },
                "overall_score": 0.7 * accuracy + 0.3 * (research_score / 100)
            }
            
            # Send results
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=eval_results))],
                name="evaluation_results",
                description="Complete evaluation results"
            )
            
            await updater.complete(
                message=new_agent_text_message(
                    f"✓ Evaluation complete! Accuracy: {accuracy:.2%}, F1: {f1:.3f}, "
                    f"Research: {research_score}/100"
                )
            )
            
        except Exception as e:
            import traceback
            await updater.failed(new_agent_text_message(
                f"Evaluation failed: {str(e)}\n{traceback.format_exc()}"
            ))
    
    def _extract_text(self, artifact) -> str:
        """Extract text content from artifact."""
        for part in artifact.parts:
            p = part.root if hasattr(part, 'root') else part
            if hasattr(p, 'text'):
                return p.text
            elif hasattr(p, 'data') and isinstance(p.data, str):
                return p.data
        return ""
    
    def _score_research(self, research_text: str) -> int:
        """Simple heuristic research scoring (0-100)."""
        score = 30  # Base score for having research
        
        if len(research_text) > 300:
            score += 20
        if len(research_text) > 500:
            score += 10
        
        # Check for key terms
        key_terms = [
            'model', 'architecture', 'feature', 'engineering',
            'cross-domain', 'transfer', 'learning', 'baseline',
            'evaluation', 'metric', 'approach', 'pipeline'
        ]
        
        found_terms = sum(1 for term in key_terms if term.lower() in research_text.lower())
        score += min(30, found_terms * 3)
        
        # Check for structure
        if any(marker in research_text for marker in ['1.', '2.', '3.', '-', '*']):
            score += 10
        
        return min(100, score)
