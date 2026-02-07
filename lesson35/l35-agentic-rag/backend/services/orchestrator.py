from models.pipeline import PipelineState, PipelineStage, StageResult
from services.budget_manager import AgentBudgetAllocator, IterationController
from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.validator_agent import ValidatorAgent
from agents.synthesizer_agent import SynthesizerAgent
from typing import Optional
import asyncio

class RAGOrchestrator:
    """Orchestrates multi-agent RAG pipeline with budget controls"""
    
    def __init__(self, api_key: str, max_tokens: int = 3000, max_iterations: int = 5):
        self.budget = AgentBudgetAllocator(max_tokens)
        self.iteration_control = IterationController(max_iterations)
        
        # Initialize agents
        self.planner = PlannerAgent(api_key)
        self.retriever = RetrieverAgent()
        self.validator = ValidatorAgent(api_key, threshold=0.7)
        self.synthesizer = SynthesizerAgent(api_key)
        
        self.state: Optional[PipelineState] = None
        self.callbacks = []  # WebSocket callbacks
    
    def register_callback(self, callback):
        """Register callback for state updates"""
        self.callbacks.append(callback)
    
    async def _notify_state_change(self):
        """Notify all callbacks of state change"""
        if self.state and self.callbacks:
            state_dict = self.state.to_dict()
            for callback in self.callbacks:
                try:
                    await callback(state_dict)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    async def process_query(self, query: str) -> PipelineState:
        """Execute full agentic RAG pipeline"""
        self.state = PipelineState(query=query)
        self.iteration_control.reset()
        
        try:
            # Stage 1: Planning
            self.state.stage = PipelineStage.PLANNING
            await self._notify_state_change()
            
            plan_result = await self.planner.analyze(
                query, 
                self.budget.get_remaining("planner")
            )
            
            if not plan_result.success:
                self.state.stage = PipelineStage.FAILED
                self.state.errors.append(f"Planning failed: {plan_result.error}")
                await self._notify_state_change()
                return self.state
            
            self.state.plan = plan_result.data
            self.state.total_tokens += plan_result.tokens_used
            self.budget.reserve("planner", plan_result.tokens_used)
            
            # Stage 2: Retrieval
            self.state.stage = PipelineStage.RETRIEVING
            await self._notify_state_change()
            
            retrieval_result = await self.retriever.retrieve(
                self.state.plan,
                self.budget.get_remaining("retriever")
            )
            
            if not retrieval_result.success:
                self.state.stage = PipelineStage.FAILED
                self.state.errors.append(f"Retrieval failed: {retrieval_result.error}")
                await self._notify_state_change()
                return self.state
            
            self.state.documents = retrieval_result.data["documents"]
            self.state.total_tokens += retrieval_result.tokens_used
            self.budget.reserve("retriever", retrieval_result.tokens_used)
            
            # Stage 3: Validation
            self.state.stage = PipelineStage.VALIDATING
            await self._notify_state_change()
            
            validation_result = await self.validator.validate(
                self.state.documents,
                query,
                self.budget.get_remaining("validator")
            )
            
            if not validation_result.success:
                # Validation failure - could retry with different params
                self.state.stage = PipelineStage.FAILED
                self.state.errors.append(f"Validation failed: {validation_result.error}")
                await self._notify_state_change()
                return self.state
            
            self.state.validation_scores = validation_result.data
            self.state.total_tokens += validation_result.tokens_used
            self.budget.reserve("validator", validation_result.tokens_used)
            
            # Stage 4: Synthesis
            self.state.stage = PipelineStage.SYNTHESIZING
            await self._notify_state_change()
            
            synthesis_result = await self.synthesizer.synthesize(
                self.state.documents,
                query,
                self.state.validation_scores,
                self.budget.get_remaining("synthesizer")
            )
            
            if not synthesis_result.success:
                self.state.stage = PipelineStage.FAILED
                self.state.errors.append(f"Synthesis failed: {synthesis_result.error}")
                await self._notify_state_change()
                return self.state
            
            self.state.response = synthesis_result.data["response"]
            self.state.total_tokens += synthesis_result.tokens_used
            self.budget.reserve("synthesizer", synthesis_result.tokens_used)
            
            # Completed
            self.state.stage = PipelineStage.COMPLETED
            self.state.metadata = {
                "budget_summary": self.budget.get_usage_summary(),
                "iterations_used": self.iteration_control.current_iteration
            }
            await self._notify_state_change()
            
        except Exception as e:
            self.state.stage = PipelineStage.FAILED
            self.state.errors.append(f"Unexpected error: {str(e)}")
            await self._notify_state_change()
        
        return self.state
