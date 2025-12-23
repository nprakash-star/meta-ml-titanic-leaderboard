"""
Meta-ML Solver Agent
Conducts research with Gemini, generates sklearn code, trains iteratively.
"""
import pandas as pd
import io
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, DataPart, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from solver_logic import train_model_async


class Agent:
    def __init__(self):
        self.model = None
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main agent logic - receives task, trains model, returns predictions."""
        
        # DEBUG: Log what we receive
        import json
        debug_info = {
            "has_metadata": hasattr(message, 'metadata'),
            "metadata": message.metadata if hasattr(message, 'metadata') else None,
            "num_parts": len(message.parts) if hasattr(message, 'parts') else 0,
            "part_types": []
        }
        
        if hasattr(message, 'parts'):
            for i, part in enumerate(message.parts):
                p = part.root if hasattr(part, 'root') else part
                part_info = {
                    "index": i,
                    "has_root": hasattr(part, 'root'),
                    "type": type(p).__name__,
                    "is_DataPart": isinstance(p, DataPart),
                    "is_TextPart": isinstance(p, TextPart),
                }
                if isinstance(p, DataPart):
                    part_info["has_data"] = hasattr(p, 'data')
                    if hasattr(p, 'data'):
                        part_info["data_keys"] = list(p.data.keys()) if isinstance(p.data, dict) else "not_dict"
                debug_info["part_types"].append(part_info)
        
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(f"DEBUG: Received message: {json.dumps(debug_info, indent=2)}")
        )
        
        # Parse task
        task_desc = "Train a machine learning model on the provided data"
        target_col = None
        train_df = None
        test_df = None
        
        if hasattr(message, 'metadata') and message.metadata:
            task_desc = message.metadata.get('task_description', task_desc)
            target_col = message.metadata.get('target_column')
            
            # Try to get CSV data from metadata (base64 encoded)
            train_csv_b64 = message.metadata.get('train_csv_base64')
            test_csv_b64 = message.metadata.get('test_csv_base64')
            
            if train_csv_b64:
                import base64
                train_csv = base64.b64decode(train_csv_b64).decode()
                train_df = pd.read_csv(io.StringIO(train_csv))
            
            if test_csv_b64:
                import base64
                test_csv = base64.b64decode(test_csv_b64).decode()
                test_df = pd.read_csv(io.StringIO(test_csv))
        
        if train_df is None:
            await updater.failed(new_agent_text_message(f"No training data provided. DEBUG: {json.dumps(debug_info, indent=2)}"))
            return
        
        # Phase 1: Research
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                "Phase 1/3: Conducting research with Gemini + Google Search..."
            )
        )
        
        # Phase 2: Training
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                "Phase 2/3: Generating model code and training..."
            )
        )
        
        try:
            result = await train_model_async(
                task_desc=task_desc,
                train_df=train_df,
                test_df=test_df if test_df is not None else train_df.iloc[:10],
                target_col=target_col
            )
            
            # Phase 3: Results
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"Phase 3/3: Training complete. Sending results..."
                )
            )
            
            # Send research artifact
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=result['research']))],
                name="research_report",
                description="Research findings from Gemini + Google Search"
            )
            
            # Send predictions
            if result['predictions']:
                predictions_csv = "prediction\n" + "\n".join(str(p) for p in result['predictions'])
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=predictions_csv))],
                    name="predictions",
                    description="Model predictions"
                )
            
            # Send metadata
            await updater.add_artifact(
                parts=[Part(root=DataPart(data={
                    "model": result['selected_model'],
                    "validation_report": result['validation_report'],
                    "problem_type": result['problem_type']
                }))],
                name="model_info",
                description="Training metadata"
            )
            
            await updater.complete(
                message=new_agent_text_message(
                    f"✓ Training complete! Model: {result['selected_model']}, "
                    f"Type: {result['problem_type']}"
                )
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            await updater.failed(new_agent_text_message(error_msg))
