from debug_gym.agents.debug_agent import Debug_5_Agent, DebugAgent
from debug_gym.agents.rewrite_agent import RewriteAgent
from debug_gym.agents.solution_agent import AgentSolution

# Conditionally import RAGAgent only if retrieval service is available
try:
    from debug_gym.agents.rag_agent import RAGAgent
except ImportError:
    # RAGAgent is not available if retrieval service is not installed
    RAGAgent = None
