"""Local Agent Demo for HairyGraph.

Demonstrates how an external AI agent or client application integrates with
HairyGraph: querying memory context before generating a response, and saving
the interaction afterward to maintain evolving conversational memory.
"""

import os
import sys
from typing import Any, Dict, Optional
import httpx

# Configuration from environment variables
HAIRYGRAPH_URL: str = os.environ.get("HAIRYGRAPH_URL", "http://127.0.0.1:8000").rstrip("/")
MEMORY_API_KEY: Optional[str] = os.environ.get("MEMORY_API_KEY", None)


class HairyGraphClient:
    """HTTP Client for communicating with the HairyGraph memory service."""

    def __init__(self, base_url: str = HAIRYGRAPH_URL, api_key: Optional[str] = MEMORY_API_KEY):
        self.base_url = base_url
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=10.0)

    def check_health(self) -> bool:
        """Check if the HairyGraph service is reachable."""
        try:
            response = self.client.get("/health")
            return response.status_code == 200
        except httpx.ConnectError:
            return False

    def get_context(self, query: str, max_tokens: int = 1500) -> Dict[str, Any]:
        """Retrieve relevant past memory context before an LLM completion."""
        payload = {"query": query, "max_tokens": max_tokens}
        response = self.client.post("/get_context", json=payload)
        response.raise_for_status()
        return response.json()

    def save_memory(self, prompt: str, response: str, pin_as_core: bool = False) -> Dict[str, Any]:
        """Persist a completed dialogue turn into the hierarchical memory graph."""
        payload = {"prompt": prompt, "response": response, "pin_as_core": pin_as_core}
        res = self.client.post("/save_memory", json=payload)
        res.raise_for_status()
        return res.json()

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve current memory graph statistics."""
        response = self.client.get("/graph_stats")
        response.raise_for_status()
        return response.json()


def simulate_llm_response(prompt: str, context: str) -> str:
    """Mock LLM response generator incorporating retrieved memory context."""
    if context:
        return (
            f"Based on your previous context:\n{context}\n\n"
            f"I have recorded and accounted for your requirement: '{prompt}'."
        )
    return f"Acknowledged. I have recorded your initial input: '{prompt}'."


def run_demo():
    print("=" * 60)
    print("HairyGraph External Client & Agent Demonstration")
    print(f"Target Service URL: {HAIRYGRAPH_URL}")
    print("=" * 60)

    client = HairyGraphClient()

    # 1. Verify service health
    if not client.check_health():
        print(f"\n[ERROR] Unable to connect to HairyGraph at {HAIRYGRAPH_URL}")
        print("Please start the service in another terminal:")
        print("    uvicorn hairy_graph.main:app --host 127.0.0.1 --port 8000\n")
        sys.exit(1)

    print("[SUCCESS] Connected to HairyGraph memory service.\n")

    # 2. Simulated conversation turns
    turns = [
        (
            "Our database must use SQLite with WAL mode enabled.",
            "Understood. We will configure SQLite with WAL mode and foreign keys enabled.",
            True,  # Pin critical architectural constraint as CORE
        ),
        (
            "What deployment strategy should we adopt for staging?",
            "For staging, we recommend blue-green deployment behind an internal reverse proxy.",
            False,
        ),
        (
            "Which database engine did we agree on earlier?",
            "We agreed on SQLite with WAL mode for local zero-maintenance operation.",
            False,
        ),
    ]

    for idx, (user_prompt, assistant_resp, pin_core) in enumerate(turns, 1):
        print(f"--- Turn {idx} ---")
        print(f"User: {user_prompt}")

        # Step A: Query memory context before inference
        print("-> Querying HairyGraph /get_context...")
        ctx_data = client.get_context(query=user_prompt, max_tokens=1000)
        mode = ctx_data.get("mode")
        tokens_used = ctx_data.get("tokens_used", 0)
        context = ctx_data.get("context", "")

        print(f"   Mode: {mode} | Tokens Used: {tokens_used}")
        if context:
            print("   Retrieved Context:\n   " + context.replace("\n", "\n   "))
        else:
            print("   (No previous context retrieved)")

        # Step B: Generate Assistant response (mocked or via LLM)
        print(f"Assistant: {assistant_resp}")

        # Step C: Save interaction to HairyGraph after inference
        print("-> Saving interaction to HairyGraph /save_memory...")
        save_res = client.save_memory(prompt=user_prompt, response=assistant_resp, pin_as_core=pin_core)
        bubble_id = save_res.get("bubble_id")
        node_type = save_res.get("node_type")
        promoted = save_res.get("promoted_node_ids", [])
        archived = save_res.get("archived_node_ids", [])

        print(f"   Saved Bubble ID: {bubble_id} (Type: {node_type})")
        if promoted:
            print(f"   Auto-Promoted to CORE: {promoted}")
        if archived:
            print(f"   Archived Nodes: {archived}")
        print()

    # 3. Print final graph stats
    stats = client.get_stats()
    print("--- Memory Graph Final Statistics ---")
    print(f"Total Clusters:     {stats.get('total_clusters')}")
    print(f"Active CORE Nodes:  {stats.get('active_core')}")
    print(f"Active STATE Nodes: {stats.get('active_state')}")
    print(f"Archived Bubbles:   {stats.get('archived_bubbles')}")
    print("=" * 60)
    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
