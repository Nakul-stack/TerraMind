import asyncio
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    
from app.chatbot.router.orchestrator import orchestrator

async def main():
    print("Testing orchestrator...")
    queries = [
        "Explain early blight",
        "Can I mix Azoxystrobin with Copper?",
        "What is the treatment for powdery mildew and what precautions apply?"
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        res = await orchestrator.ask(q, {"identified_crop": "Tomato"}, top_k=3)
        print(f"Strategy: {res['strategy']}")
        print(f"Confidence: {res['confidence']}")
        print(f"Answer snippet: {res['answer'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
