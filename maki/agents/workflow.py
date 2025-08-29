from typing import Dict, List, Tuple, Literal
from ..utils.logging import logger
from .relevance_checker import RelevanceChecker
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .router import Router
from .tools import welcome_tool
from ..retriever.builder import SimpleKeywordRetriever, Document
from ..config import settings
from langgraph.graph import StateGraph, END
from ..utils import start_run, append_and_log, persist_agent_trace

State = Dict[str, object]
Domain = Literal["GYM", "CLIENTS", "UNKNOWN"]


class Workflow:
    """Orquesta con LangGraph: welcome → retrieve → relevance/route → research → verify
    → (si falla) redraft → fallback → salida.
    """

    def __init__(
        self,
        gym_retriever: SimpleKeywordRetriever,
        clients_retriever: SimpleKeywordRetriever,
        completion_fallback=True,
    ):
        self.gym_retriever = gym_retriever
        self.clients_retriever = clients_retriever
        self.relevance = RelevanceChecker()     # usa LLM/embeddings si hay
        self.router = Router()
        self.research = ResearchAgent()         # usa LLM para síntesis si hay
        self.verify = VerificationAgent()       # usa LLM para verificación si hay
        self.completion_fallback = completion_fallback  # callable(question)->str (opcional)

        # --- Definir grafo ---
        graph = StateGraph(State)

        graph.add_node("welcome", self._n_welcome)
        graph.add_node("retrieve", self._n_retrieve)
        graph.add_node("route", self._n_route)
        graph.add_node("research", self._n_research)
        graph.add_node("verify", self._n_verify)
        graph.add_node("redraft", self._n_redraft)
        graph.add_node("fallback", self._n_fallback)
        graph.add_node("format_output", self._n_format_output)

        # Secuencia base
        graph.set_entry_point("welcome")
        graph.add_edge("welcome", "retrieve")
        graph.add_edge("retrieve", "route")
        graph.add_edge("route", "research")
        graph.add_edge("research", "verify")

        # Condicional: ¿necesita redraft?
        def need_redraft(state: State) -> str:
            ver = state.get("verification", {})
            draft = (state.get("draft") or "").strip()
            if not draft:
                return "fallback"
            if ver.get("supported") == "NO" or ver.get("relevant") == "NO":
                return "redraft"
            return "ok"

        graph.add_conditional_edges(
            "verify",
            need_redraft,
            {
                "ok": "format_output",
                "redraft": "redraft",
                "fallback": "fallback",
            },
        )

        # Después de redraft vamos directo a formato (versión 1-iteración)
        graph.add_edge("redraft", "format_output")

        # Fallback final si seguimos sin draft
        graph.add_edge("fallback", "format_output")

        # Salida
        graph.add_edge("format_output", END)

        self._graph = graph.compile()

    # ------------------- NODOS -------------------
    def _n_welcome(self, state: State) -> State:
        return append_and_log(state, "tool.welcome", welcome_tool(), stage="init")

    def _n_retrieve(self, state: State) -> State:
        q: str = state["question"]
        gyms_docs = self.gym_retriever.invoke(q, k=4)
        clients_docs = self.clients_retriever.invoke(q, k=4)
        combined = gyms_docs + clients_docs
        state = append_and_log(state, "retrieve.counts",
                            f"gyms={len(gyms_docs)}, clients={len(clients_docs)}",
                            stage="retrieve")
        return {**state, "gyms_docs": gyms_docs, "clients_docs": clients_docs, "combined": combined}


    def _n_route(self, state: State) -> State:
        q: str = state["question"]
        combined: List[Document] = state["combined"]  # type: ignore
        label = self.relevance.classify(q, combined)
        domain: Domain = self.router.route(label)
        state = append_and_log(state, "route.label", label, stage="route")
        state = append_and_log(state, "route.domain", domain, stage="route", domain=domain)
        return {**state, "label": label, "domain": domain}

    def _n_research(self, state: State) -> State:
        q: str = state["question"]
        domain: Domain = state.get("domain", "UNKNOWN")  # type: ignore
        gyms_docs: List[Document] = state.get("gyms_docs", [])  # type: ignore
        clients_docs: List[Document] = state.get("clients_docs", [])  # type: ignore

        if domain == "GYM":
            docs = gyms_docs
        elif domain == "CLIENTS":
            docs = clients_docs
        else:
            docs = []

        res = self.research.run(q, docs)
        draft = res.get("answer", "")
        evidence = res.get("evidence", "")
        state = append_and_log(state, "research.draft", draft, stage="research", domain=domain)
        return {**state, "draft": draft, "evidence": evidence}



    def _n_verify(self, state: State) -> State:
        q: str = state["question"]
        draft: str = state.get("draft", "")  # type: ignore
        ver = self.verify.run(q, draft)
        state = append_and_log(state, "verify.report", str(ver), stage="verify",
                            domain=state.get("domain", "UNKNOWN"))
        return {**state, "verification": ver}


    def _n_redraft(self, state: State) -> State:
        q: str = state["question"]
        domain: Domain = state.get("domain", "UNKNOWN")  # type: ignore
        gyms_docs: List[Document] = state.get("gyms_docs", [])  # type: ignore
        clients_docs: List[Document] = state.get("clients_docs", [])  # type: ignore

        more_docs = (clients_docs if domain == "GYM" else gyms_docs)[:2]
        base_docs = gyms_docs if domain == "GYM" else clients_docs if domain == "CLIENTS" else []
        res2 = self.research.run(q, base_docs + more_docs)
        redraft = res2.get("answer", "")
        state = append_and_log(state, "research.redraft", redraft, stage="research", domain=domain)
        return {**state, "draft": redraft}


    def _n_fallback(self, state: State) -> State:
        q: str = state["question"]
        draft: str = state.get("draft", "")  # type: ignore

        if draft.strip():
            return state  # ya hay respuesta

        # no hay draft: intentamos completion
        if self.completion_fallback:
            try:
                gen = self.completion_fallback(q)  # puede lanzar o devolver ""
                if gen and gen.strip():
                    final = f"{settings.IRRELEVANT_MSG}\n\n{gen.strip()}"
                    state = append_and_log(state, "fallback", "completion.ok", stage="fallback",
                                        domain=state.get("domain", "UNKNOWN"))
                    return {**state, "draft": final}
                else:
                    state = append_and_log(state, "fallback", "completion.empty", stage="fallback",
                                        domain=state.get("domain", "UNKNOWN"))
            except Exception as e:
                # registramos el error y seguimos con mensaje
                state = append_and_log(state, "fallback", f"completion.error: {type(e).__name__}: {e}", stage="fallback",
                                    domain=state.get("domain", "UNKNOWN"))

        # sin completion o falló → mensaje por defecto
        final = settings.IRRELEVANT_MSG
        state = append_and_log(state, "fallback", "message", stage="fallback",
                            domain=state.get("domain", "UNKNOWN"))
        return {**state, "draft": final}



    def _n_format_output(self, state: State) -> State:
        # Normaliza la salida
        return state

    def run(self, question: str) -> Dict:
        """Ejecuta el grafo"""
        start_run()
        initial: State = {
            "question": question,
            "trace": [],  # List[Tuple[str, str]]
        }
        final_state: State = self._graph.invoke(initial)  # ejecución sincrónica simple
        draft = final_state.get("draft", "") or ""
        domain = final_state.get("domain", "UNKNOWN")
        trace = final_state.get("trace", [])
        persist_agent_trace(final_state, domain=domain)
        return {
            "final_answer": draft,
            "trace": trace,
            "domain": domain,
        }
