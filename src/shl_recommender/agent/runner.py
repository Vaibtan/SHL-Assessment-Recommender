"""Top-level agent orchestration — features → router → handler dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass

from shl_recommender.agent.handlers._base import HandlerResult
from shl_recommender.agent.handlers.clarify import handle_clarify
from shl_recommender.agent.handlers.compare import handle_compare
from shl_recommender.agent.handlers.recommend import handle_recommend
from shl_recommender.agent.handlers.refine import handle_refine
from shl_recommender.agent.handlers.refuse import handle_refuse
from shl_recommender.agent.llm import LLMClient, begin_llm_stats, end_llm_stats
from shl_recommender.agent.router import Intent, RouterDecision, route, RouteRequest
from shl_recommender.assembly.reply import assemble_chat_response
from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.features.pipeline import FeatureBundle, build_feature_bundle
from shl_recommender.schemas import ChatRequest, ChatResponse, Message


@dataclass(slots=True)
class AgentResult:
    """Internal handler-level result before final assembly."""

    response: ChatResponse
    decision: RouterDecision
    features: FeatureBundle
    handler_result: HandlerResult
    timings: dict[str, int]
    llm_stats: dict[str, object]


class Agent:
    """Composes the policy → planning → safety pipeline.

    Constructed once per process; safe to call concurrently.
    """

    def __init__(
        self,
        index: CatalogIndex,
        llm: LLMClient,
    ) -> None:
        self.index = index
        self.llm = llm

    async def chat(self, payload: ChatRequest) -> AgentResult:
        timings: dict[str, int] = {}
        llm_token = begin_llm_stats()
        t0 = time.perf_counter()
        try:
            features = await build_feature_bundle(payload.messages, self.index, self.llm)
            timings["features_ms"] = int((time.perf_counter() - t0) * 1000)

            t1 = time.perf_counter()
            decision = await route(RouteRequest(payload.messages, features), self.llm)
            timings["router_ms"] = int((time.perf_counter() - t1) * 1000)

            t2 = time.perf_counter()
            result = await self._dispatch(payload.messages, decision, features)
            timings["handler_ms"] = int((time.perf_counter() - t2) * 1000)

            t3 = time.perf_counter()
            response = assemble_chat_response(
                reply_text=result.reply_text,
                entity_ids=result.entity_ids,
                is_final_turn_flag=decision.is_final_turn,
                index=self.index,
            )
            timings["assembly_ms"] = int((time.perf_counter() - t3) * 1000)

            timings["total_ms"] = int((time.perf_counter() - t0) * 1000)
            return AgentResult(
                response=response,
                decision=decision,
                features=features,
                handler_result=result,
                timings=timings,
                llm_stats=end_llm_stats(llm_token),
            )
        except BaseException:
            end_llm_stats(llm_token)
            raise

    async def _dispatch(
        self,
        messages: list[Message],
        decision: RouterDecision,
        features: FeatureBundle,
    ) -> HandlerResult:
        if decision.intent == Intent.REFUSE:
            return await handle_refuse(decision=decision)
        if decision.intent == Intent.CLARIFY:
            return await handle_clarify(
                messages=messages, decision=decision, features=features, llm=self.llm
            )
        if decision.intent == Intent.COMPARE:
            return await handle_compare(
                messages=messages,
                decision=decision,
                features=features,
                llm=self.llm,
                index=self.index,
            )
        if decision.intent == Intent.REFINE:
            return await handle_refine(
                messages=messages,
                decision=decision,
                features=features,
                llm=self.llm,
                index=self.index,
            )
        # Default: recommend
        return await handle_recommend(
            messages=messages,
            decision=decision,
            features=features,
            llm=self.llm,
            index=self.index,
        )
