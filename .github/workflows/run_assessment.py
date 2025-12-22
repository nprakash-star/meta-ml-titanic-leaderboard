#!/usr/bin/env python3
"""
Assessment runner using A2A Python SDK
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart, DataPart
from uuid import uuid4


def create_message(text: str, metadata: dict) -> Message:
    """Create an A2A message with metadata"""
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        metadata=metadata
    )


async def run_assessment(green_agent_url: str, solver_url: str, metadata: dict):
    """Run the assessment by calling the green agent"""
    
    print(f"Connecting to green agent at {green_agent_url}...")
    
    async with httpx.AsyncClient(timeout=600) as httpx_client:
        # Get agent card
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=green_agent_url)
        agent_card = await resolver.get_agent_card()
        print(f"Connected to: {agent_card.name}")
        
        # Create client
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        
        # Create evaluation request message
        msg = create_message(
            text="Please evaluate the solver agent on Titanic survival prediction",
            metadata=metadata
        )
        
        print("Sending evaluation request...")
        
        # Send message and get response
        last_event = None
        async for event in client.send_message(msg):
            last_event = event
        
        # Extract results
        if not last_event:
            raise RuntimeError("No response from green agent")
        
        results = None
        
        # Handle different response types
        if isinstance(last_event, Message):
            # Direct message response
            for part in last_event.parts:
                if isinstance(part.root, DataPart):
                    results = part.root.data
                    break
        elif isinstance(last_event, tuple):
            # Task response
            task, update = last_event
            
            # Check for artifacts
            if task.artifacts:
                for artifact in task.artifacts:
                    if artifact.name == "evaluation_results":
                        for part in artifact.parts:
                            if isinstance(part.root, DataPart):
                                results = part.root.data
                                break
            
            # If no artifacts, check message
            if not results and task.status.message:
                for part in task.status.message.parts:
                    if isinstance(part.root, DataPart):
                        results = part.root.data
                        break
        
        if not results:
            print("ERROR: Could not extract evaluation results from response", file=sys.stderr)
            print(f"Response: {last_event}", file=sys.stderr)
            sys.exit(1)
        
        return results


async def main():
    if len(sys.argv) < 2:
        print("Usage: run_assessment.py <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # Assessment configuration
    green_agent_url = "http://localhost:9010"
    solver_url = "http://solver:9009"
    
    metadata = {
        "agent_url": solver_url,
        "train_data_url": "https://raw.githubusercontent.com/nprakash-star/meta-ml-titanic-leaderboard/main/data/titanic_train_80.csv",
        "test_data_url": "https://raw.githubusercontent.com/nprakash-star/meta-ml-titanic-leaderboard/main/data/titanic_test_20_features.csv",
        "test_labels_url": "https://raw.githubusercontent.com/nprakash-star/meta-ml-titanic-leaderboard/main/data/titanic_test_20_labels.csv",
        "target_column": "Survived",
        "task_description": "Predict Titanic passenger survival based on demographics and ticket information"
    }
    
    try:
        results = await run_assessment(green_agent_url, solver_url, metadata)
        
        # Format for AgentBeats
        output_data = {
            "participants": {
                "agent": "019b4528-623c-7202-97d8-33bb09081a86"
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "commit": "${GITHUB_SHA}",
            "run_id": "${GITHUB_RUN_ID}",
            "results": [results]
        }
        
        # Save results
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        # Print summary
        print(f"\n✓ Successfully completed assessment")
        print(f"  Accuracy: {results['performance']['accuracy']:.2%}")
        print(f"  F1 Score: {results['performance']['f1_score']:.3f}")
        print(f"  Research Score: {results['research']['score']}/100")
        print(f"  Overall Score: {results['overall_score']:.3f}")
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
