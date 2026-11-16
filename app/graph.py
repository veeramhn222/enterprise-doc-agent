"""LangGraph orchestration of the agents.

Each node wraps one of the agent functions from agents.py; LangGraph owns the control flow:

    START -> precheck -> planner -> retriever -> reasoner -> validator --(weak answer, 1st time)--> reviser -> validator
                |                      |                                \\--(otherwise)--> finish -> END
                '-- no documents       '-- nothing relevant  ==> END (refusal, LLM never called)
"""
from typing import Any, TypedDict

from . import agents, guardrails


class AgentState(TypedDict, total=False):
    question: str
    run: Any          # agents.Run: trace + retrieved sources for this question
    steps: list       # planner output
    answer: str
    verdict: dict
    revised: bool
    final: dict       # set when the response is ready -> graph ends


# ----------------------------------------------------------------- nodes
def precheck_node(s):
    early = agents._no_docs(s["run"])
    return {"final": early} if early else {}


def planner_node(s):
    return {"steps": agents.planner_agent(s["run"])}


def retriever_node(s):
    agents.retriever_agent(s["run"], s["steps"])
    early = agents._nothing_found(s["run"])
    return {"final": early} if early else {}


def reasoner_node(s):
    return {"answer": agents.reasoner_agent(s["run"])}


def validator_node(s):
    return {"verdict": agents.validator_agent(s["run"], s["answer"])}


def reviser_node(s):
    return {"answer": agents._revise(s["run"], s["verdict"]), "revised": True}


def finish_node(s):
    return {"final": agents._finalize(s["run"], s["answer"], s["verdict"], bool(s.get("revised")))}


# ---------------------------------------------------------------- routing
def route_early_exit(s):
    return "end" if s.get("final") else "continue"


def route_after_validation(s):
    if not s.get("revised") and agents._needs_revision(s["verdict"], s["run"]):
        return "revise"
    return "finish"


_compiled = None


def get_graph():
    """Build and compile the state machine once. Raises ImportError if langgraph is not installed."""
    global _compiled
    if _compiled is None:
        from langgraph.graph import END, START, StateGraph
        g = StateGraph(AgentState)
        for name, fn in [("precheck", precheck_node), ("planner", planner_node), ("retriever", retriever_node),
                         ("reasoner", reasoner_node), ("validator", validator_node),
                         ("reviser", reviser_node), ("finish", finish_node)]:
            g.add_node(name, fn)
        g.add_edge(START, "precheck")
        g.add_conditional_edges("precheck", route_early_exit, {"end": END, "continue": "planner"})
        g.add_edge("planner", "retriever")
        g.add_conditional_edges("retriever", route_early_exit, {"end": END, "continue": "reasoner"})
        g.add_edge("reasoner", "validator")
        g.add_conditional_edges("validator", route_after_validation, {"revise": "reviser", "finish": "finish"})
        g.add_edge("reviser", "validator")
        g.add_edge("finish", END)
        _compiled = g.compile()
    return _compiled


def run(question: str) -> dict:
    q = guardrails.validate_question(question)
    state_run = agents.Run(q)
    state_run.log("Orchestrator", "LangGraph: precheck > planner > retriever > reasoner > validator > (reviser > validator) > finish")
    out = get_graph().invoke({"question": q, "run": state_run, "revised": False})
    return out["final"]
