import time
import structlog
from typing import List, Dict, Any
from backend.models.synthesis_models import (
    SynthesisRequest, SynthesisResponse, SynthesisState,
    ReasoningStep, Citation, SynthesisEvent
)
from backend.agents.cot_engine import CoTEngine
from backend.agents.citation_mapper import CitationMapper
from backend.agents.coherence_validator import CoherenceValidator
from backend.agents.xai_formatter import XAIFormatter

logger = structlog.get_logger()

class SynthesizerAgent:
    def __init__(self, api_key: str, model_name: str, citation_granularity: str = "medium"):
        self.cot_engine = CoTEngine(api_key, model_name)
        self.citation_mapper = CitationMapper(api_key, model_name, citation_granularity)
        self.coherence_validator = CoherenceValidator(api_key, model_name)
        self.xai_formatter = XAIFormatter()
        self.state = SynthesisState.AWAITING_CONTEXT
        
    def synthesize(self, request: SynthesisRequest, trace_id: str = None) -> SynthesisResponse:
        """Main synthesis orchestration"""
        start_time = time.time()
        trace_id = trace_id or f"synth_{int(time.time() * 1000)}"
        
        logger.info("synthesis_started", trace_id=trace_id, query=request.query[:100])
        
        # State: PLANNING_SYNTHESIS
        self.state = SynthesisState.PLANNING_SYNTHESIS
        reasoning_plan = self.cot_engine.plan_reasoning(request.query, request.validated_chunks)
        logger.info("reasoning_planned", trace_id=trace_id, steps=len(reasoning_plan))
        
        # State: GENERATING_CLAIMS
        self.state = SynthesisState.GENERATING_CLAIMS
        reasoning_chain = []
        previous_outputs = []
        
        for step_def in reasoning_plan:
            step_result = self.cot_engine.execute_step(
                step_def, request.query, request.validated_chunks, previous_outputs
            )
            reasoning_chain.append(step_result)
            previous_outputs.append(step_result.output_data)
            
            # Log step completion
            event = SynthesisEvent(
                step_name=step_result.step_name,
                inputs=step_result.input_data,
                outputs=step_result.output_data,
                confidence=step_result.confidence,
                latency_ms=step_result.latency_ms,
                trace_id=trace_id
            )
            logger.info("synthesis_step_complete", event=event.model_dump())
        
        # State: CLAIMS_COMPLETE
        self.state = SynthesisState.CLAIMS_COMPLETE
        final_output = reasoning_chain[-1].output_data.get("output", "")
        
        # State: MAPPING_CITATIONS
        self.state = SynthesisState.MAPPING_CITATIONS
        citations = self.citation_mapper.map_claims_to_sources(final_output, request.validated_chunks)
        logger.info("citations_mapped", trace_id=trace_id, citation_count=len(citations))
        
        # State: ASSEMBLING_RESPONSE
        self.state = SynthesisState.ASSEMBLING_RESPONSE
        response_with_citations = self.citation_mapper.inject_citations(final_output, citations)
        
        # State: VALIDATING_OUTPUT
        self.state = SynthesisState.VALIDATING_OUTPUT
        is_valid, validation_results = self.coherence_validator.validate(
            response_with_citations, citations, request.validated_chunks
        )
        
        if not is_valid:
            logger.warning("validation_failed", trace_id=trace_id, results=validation_results)
            self.state = SynthesisState.REGENERATE_REQUIRED
            # In production, would trigger retry with adjusted parameters
        else:
            self.state = SynthesisState.READY_TO_SEND
        
        # Compute overall confidence
        confidences = [step.confidence for step in reasoning_chain]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        synthesis_time_ms = (time.time() - start_time) * 1000
        
        logger.info("synthesis_complete", 
                   trace_id=trace_id, 
                   latency_ms=synthesis_time_ms,
                   confidence=overall_confidence)
        
        return SynthesisResponse(
            response_text=response_with_citations,
            citations=citations,
            reasoning_chain=reasoning_chain,
            overall_confidence=overall_confidence,
            synthesis_time_ms=synthesis_time_ms,
            state=self.state,
            metadata={
                "trace_id": trace_id,
                "validation_results": validation_results,
                "step_count": len(reasoning_chain)
            }
        )
