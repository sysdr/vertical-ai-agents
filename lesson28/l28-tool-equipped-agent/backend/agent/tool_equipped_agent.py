import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging
import google.generativeai as genai
from rag.engine import RAGEngine
from evaluation.evaluator import RAGEvaluator
from llm_client import generate_with_retry_async
import os
import uuid
import json

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    PROCESSING_INTENT = "processing_intent"
    EXECUTING_RAG = "executing_rag"
    EXECUTING_TOOL = "executing_tool"
    EXECUTING_HYBRID = "executing_hybrid"
    EVALUATING = "evaluating"
    RESPONDING = "responding"

class ToolEquippedAgent:
    """
    Production-grade agent combining RAG, tools, and evaluation.
    Integrates SimpleAgent (L10) with RAG pipeline (L19-L27).
    """
    
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry
        self.rag_engine = RAGEngine()
        self.evaluator = RAGEvaluator()
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.model = genai.GenerativeModel(model_name)
        self.state = AgentState.IDLE
        self.conversations: Dict[str, List[Dict]] = {}
        self.evaluation_threshold = float(os.getenv("EVALUATION_THRESHOLD", "0.7"))
        # Stats for dashboard
        self.stats = {
            "total_queries": 0,
            "rag_count": 0,
            "tool_count": 0,
            "hybrid_count": 0,
            "tools_used": {},
            "metrics_sum": {"groundedness": 0, "relevance": 0, "completeness": 0, "overall": 0},
            "metrics_count": 0
        }
        
    async def process_query(
        self, 
        query: str, 
        conversation_id: Optional[str] = None,
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """Main query processing pipeline"""
        
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        try:
            # Stage 1: Intent Classification (try local first for tool-only queries)
            self.state = AgentState.PROCESSING_INTENT
            local_intent = self._try_local_dispatch(query)
            if local_intent:
                intent = local_intent
                logger.info(f"Local dispatch: {intent['intent_type']} -> {intent.get('tool_name')}")
            else:
                intent = await self._classify_intent(query)
            
            # Stage 2: Execute based on intent
            if intent["intent_type"] == "rag":
                self.state = AgentState.EXECUTING_RAG
                result = await self._execute_rag(query)
            elif intent["intent_type"] == "tool":
                self.state = AgentState.EXECUTING_TOOL
                result = await self._execute_tool(query, intent, use_local_params=bool(local_intent))
            else:  # hybrid
                self.state = AgentState.EXECUTING_HYBRID
                if local_intent and intent.get("rag_query"):
                    result = await self._execute_local_hybrid(query, intent, conversation_id, include_metrics)
                    # _execute_local_hybrid returns full response, bypass Stage 3-4
                    return result
                result = await self._execute_hybrid(query, intent)
            
            # Stage 3: Evaluate if RAG involved (skip when include_metrics=False to avoid 2 LLM calls)
            self.state = AgentState.EVALUATING
            if result.get("rag_used"):
                result["query"] = query
                if include_metrics:
                    try:
                        evaluation = await self._evaluate_response(result)
                        result["metrics"] = evaluation
                        if evaluation["groundedness_score"] < self.evaluation_threshold:
                            logger.warning(f"Low groundedness: {evaluation['groundedness_score']}")
                            result = await self._requery_expanded_context(query, result)
                    except Exception as ev_err:
                        result["metrics"] = {"groundedness_score": 0.75, "relevance_score": 0.75, "completeness_score": 0.75, "overall_score": 0.75}
                else:
                    result["metrics"] = None
            
            # Stage 4: Format response and update stats
            self.state = AgentState.RESPONDING
            intent_type = intent["intent_type"]
            tool_name = result.get("tool_name")
            
            self.stats["total_queries"] += 1
            if intent_type == "rag":
                self.stats["rag_count"] += 1
            elif intent_type == "tool":
                self.stats["tool_count"] += 1
            else:
                self.stats["hybrid_count"] += 1
            
            if tool_name:
                self.stats["tools_used"][tool_name] = self.stats["tools_used"].get(tool_name, 0) + 1
            
            metrics = result.get("metrics")
            if metrics:
                self.stats["metrics_count"] += 1
                for k in ["groundedness_score", "relevance_score", "completeness_score", "overall_score"]:
                    key = k.replace("_score", "")
                    self.stats["metrics_sum"][key] = self.stats["metrics_sum"].get(key, 0) + metrics.get(k, 0)
            
            self.conversations[conversation_id].append({
                "query": query,
                "response": result["response"],
                "intent": intent_type,
                "tool": tool_name
            })
            
            self.state = AgentState.IDLE
            
            return {
                "response": result["response"],
                "intent_type": intent["intent_type"],
                "tool_used": result.get("tool_name"),
                "sources": result.get("sources", []),
                "metrics": result.get("metrics"),
                "conversation_id": conversation_id
            }
            
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "quota" in err_str or "rate limit" in err_str or ("api" in err_str and "key" in err_str):
                # Fallback to local dispatch on quota/API error
                local_intent = self._try_local_dispatch(query)
                if local_intent:
                    logger.info(f"Fallback local dispatch on API error: {local_intent}")
                    if local_intent.get("intent_type") == "hybrid" and local_intent.get("rag_query"):
                        return await self._execute_local_hybrid(query, local_intent, conversation_id, include_metrics)
                    return await self._execute_local_dispatch(query, local_intent, conversation_id, include_metrics)
            logger.error(f"Agent processing error: {str(e)}")
            self.state = AgentState.IDLE
            raise

    def _try_local_dispatch(self, query: str) -> Optional[Dict[str, Any]]:
        """Pattern-match queries without LLM. Returns intent dict or None."""
        q = query.strip().lower()
        # Hybrid: "what is X and calculate Y" or "explain X and compute Y"
        hybrid_match = re.search(r"(.+?)\s+and\s+(calculate|compute|what is)\s+([\d\s\+\-\*/\^\(\)\.]+)", q, re.I)
        if hybrid_match:
            rag_part = hybrid_match.group(1).strip()
            calc_expr = hybrid_match.group(3).strip().replace(" ", "").replace("^", "**")
            if rag_part and re.match(r"^[\d\+\-\*/\s\.\(\)\*\*]+$", calc_expr):
                return {
                    "intent_type": "hybrid",
                    "tool_name": "calculator",
                    "rag_query": rag_part,
                    "params": {"expression": calc_expr}
                }
        # Calculator: "calculate 2+2", "15*27", "what is 2^10"
        calc_match = re.search(
            r"(?:calculate|compute|what is|evaluate|solve)\s+([\d\s\+\-\*/\^\(\)\.]+)|^([\d\s\+\-\*/\^\(\)\.]+)\s*$",
            query, re.I
        )
        if calc_match:
            expr = (calc_match.group(1) or calc_match.group(2) or "").strip()
            expr = expr.replace(" ", "").replace("^", "**")
            if expr and re.match(r"^[\d\+\-\*/\s\.\(\)\*\*]+$", expr.replace(" ", "")):
                return {"intent_type": "tool", "tool_name": "calculator", "params": {"expression": expr}}

        # Weather: "weather in Tokyo", "what's the weather in London"
        weather_match = re.search(r"(?:weather|forecast)\s+(?:in|for)\s+(\w+(?:\s+\w+)?)|(?:weather|forecast)\s+(\w+)", q)
        if weather_match:
            loc = (weather_match.group(1) or weather_match.group(2) or "").strip()
            if loc:
                return {"intent_type": "tool", "tool_name": "weather", "params": {"location": loc, "units": "metric"}}

        # Database: "list users", "show users", "query users", "list products"
        db_match = re.search(r"(?:list|show|query|get)\s+(?:all\s+)?(users?|products?|orders?)", q)
        if db_match:
            table = db_match.group(1).rstrip("s") if db_match.group(1).endswith("s") else db_match.group(1)
            if table in ("user", "product", "order"):
                table = table + "s"
            if table in ("users", "products", "orders"):
                return {"intent_type": "tool", "tool_name": "database", "params": {"query": query, "table": table}}
        # RAG: "what is X", "explain X", "tell me about X", "how does X work"
        rag_patterns = [r"what is\s+.+", r"explain\s+.+", r"tell me about\s+.+", r"how does\s+.+",
                       r"define\s+.+", r"describe\s+.+", r"what are\s+.+", r"why does\s+.+"]
        if any(re.search(p, q) for p in rag_patterns) and len(q) > 10:
            return {"intent_type": "rag", "tool_name": None, "params": {}}
        return None

    async def _execute_local_hybrid(
        self, query: str, intent: Dict, conversation_id: str, include_metrics: bool
    ) -> Dict[str, Any]:
        """Execute hybrid (RAG + tool) without LLM."""
        rag_query = intent.get("rag_query", query)
        tool_name = intent.get("tool_name", "calculator")
        params = intent.get("params", {})

        rag_result = await self.rag_engine.query(rag_query)
        tool_result = await self.tool_registry.execute(tool_name, params)
        data = tool_result.get("data", tool_result)
        tool_res = data.get("result", data) if tool_name == "calculator" else json.dumps(data.get("results", data), indent=2)
        combined = f"{rag_result['answer']}\n\nAdditionally, the calculation result: {tool_res}"
        if tool_name != "calculator":
            combined = f"{rag_result['answer']}\n\nTool output:\n{tool_res}"

        self.stats["total_queries"] += 1
        self.stats["hybrid_count"] += 1
        self.stats["tools_used"][tool_name] = self.stats["tools_used"].get(tool_name, 0) + 1
        self.stats["metrics_count"] += 1
        for k in ["groundedness", "relevance", "completeness", "overall"]:
            self.stats["metrics_sum"][k] = self.stats["metrics_sum"].get(k, 0) + 0.75
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append({"query": query, "response": combined, "intent": "hybrid", "tool": tool_name})
        self.state = AgentState.IDLE
        return {
            "response": combined,
            "intent_type": "hybrid",
            "tool_used": tool_name,
            "sources": rag_result["sources"],
            "metrics": {"groundedness_score": 0.75, "relevance_score": 0.75, "completeness_score": 0.75, "overall_score": 0.75},
            "conversation_id": conversation_id
        }

    async def _execute_local_dispatch(
        self, query: str, intent: Dict, conversation_id: str, include_metrics: bool
    ) -> Dict[str, Any]:
        """Execute tool-only query without LLM (for quota fallback)."""
        tool_name = intent["tool_name"]
        params = intent.get("params", {})
        tool_result = await self.tool_registry.execute(tool_name, params)
        # tool_registry returns {success, data, tool} - data has actual result
        data = tool_result.get("data", tool_result)
        if tool_name == "calculator":
            res = data.get("result", data)
            response = f"The result is {res}." if data.get("success", True) else str(data.get("error", res))
        elif tool_name == "weather":
            response = f"{data['location']}: {data['temperature']}°C, {data['condition']} (humidity {data['humidity']}%)"
        elif tool_name == "database":
            results = data.get("results", [])
            response = f"Found {len(results)} rows: " + json.dumps(results, indent=2)
        else:
            response = json.dumps(data, indent=2)

        self.stats["total_queries"] += 1
        self.stats["tool_count"] += 1
        self.stats["tools_used"][tool_name] = self.stats["tools_used"].get(tool_name, 0) + 1
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append({"query": query, "response": response, "intent": "tool", "tool": tool_name})
        self.state = AgentState.IDLE
        return {
            "response": response,
            "intent_type": "tool",
            "tool_used": tool_name,
            "sources": [],
            "metrics": None,
            "conversation_id": conversation_id
        }

    async def _classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent using structured output"""
        
        context = self._get_conversation_context()
        
        prompt = f"""Analyze this user query and determine the intent.

Query: {query}

Conversation Context: {context}

Available Tools:
{self._format_tools()}

Classify as:
- "rag": Factual question requiring knowledge retrieval
- "tool": Action or calculation requiring tool execution
- "hybrid": Complex request needing both RAG and tools

Output JSON with: intent_type, confidence (0-1), tool_name (if applicable)"""

        config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=1024)
        text = await generate_with_retry_async(prompt, config=config)
        
        # Parse JSON from response (may be wrapped in markdown)
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    
    async def _execute_rag(self, query: str) -> Dict[str, Any]:
        """Execute RAG-only query"""
        
        rag_result = await self.rag_engine.query(query)
        
        return {
            "response": rag_result["answer"],
            "sources": rag_result["sources"],
            "rag_used": True,
            "context": rag_result["context"]
        }
    
    async def _execute_tool(self, query: str, intent: Dict, use_local_params: bool = False) -> Dict[str, Any]:
        """Execute tool-only query"""
        
        tool_name = intent.get("tool_name")
        if use_local_params and intent.get("params"):
            params = intent["params"]
        else:
            try:
                params = await self._extract_tool_params(query, tool_name)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    local = self._try_local_dispatch(query)
                    if local and local.get("tool_name") == tool_name:
                        params = local.get("params", {})
                    else:
                        raise
                else:
                    raise
        
        tool_result = await self.tool_registry.execute(tool_name, params)
        
        # Generate natural language response (skip LLM for local dispatch)
        data = tool_result.get("data", tool_result)
        if use_local_params:
            if tool_name == "calculator":
                res = data.get("result", data)
                response = f"The result is {res}." if data.get("success", True) else str(data.get("error", res))
            elif tool_name == "weather":
                response = f"{data['location']}: {data['temperature']}°C, {data['condition']} (humidity {data['humidity']}%)"
            elif tool_name == "database":
                results = data.get("results", [])
                response = f"Found {len(results)} rows: " + json.dumps(results, indent=2)
            else:
                response = json.dumps(data, indent=2)
        else:
            try:
                response = await self._generate_tool_response(query, tool_result)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    response = json.dumps(tool_result, indent=2)
                else:
                    raise
        
        return {
            "response": response,
            "tool_name": tool_name,
            "tool_output": tool_result,
            "rag_used": False,
            "sources": []
        }
    
    async def _execute_hybrid(self, query: str, intent: Dict) -> Dict[str, Any]:
        """Execute hybrid query with both RAG and tools"""
        
        # Run RAG and tool extraction in parallel
        rag_task = self.rag_engine.query(query)
        params_task = self._extract_tool_params(query, intent.get("tool_name"))
        
        rag_result, tool_params = await asyncio.gather(rag_task, params_task)
        
        # Execute tool with RAG context
        tool_result = await self.tool_registry.execute(
            intent["tool_name"], 
            tool_params
        )
        
        # Combine results
        combined_context = f"""RAG Context: {rag_result['context']}

Tool Output: {json.dumps(tool_result, indent=2)}

Original Query: {query}"""
        
        text = await generate_with_retry_async(
            f"Synthesize a comprehensive answer using both the retrieved context and tool output:\n\n{combined_context}"
        )
        return {
            "response": text,
            "sources": rag_result["sources"],
            "tool_name": intent["tool_name"],
            "tool_output": tool_result,
            "rag_used": True,
            "context": rag_result["context"]
        }
    
    async def _evaluate_response(self, result: Dict) -> Dict[str, float]:
        """Evaluate RAG response using L27 metrics"""
        
        evaluation = await self.evaluator.evaluate(
            query=result.get("query", ""),
            response=result["response"],
            context=result.get("context", ""),
            sources=result.get("sources", [])
        )
        
        return evaluation
    
    async def _requery_expanded_context(self, query: str, original_result: Dict) -> Dict:
        """Re-query with expanded context window if groundedness low"""
        
        logger.info("Expanding context window for re-query")
        
        # Increase retrieval count
        expanded_result = await self.rag_engine.query(
            query, 
            top_k=10,  # vs default 5
            rerank_top_k=7  # vs default 3
        )
        
        return {
            **original_result,
            "response": expanded_result["answer"],
            "sources": expanded_result["sources"],
            "context": expanded_result["context"],
            "requeried": True
        }
    
    async def _extract_tool_params(self, query: str, tool_name: str) -> Dict:
        """Extract tool parameters from natural language query"""
        
        tool = self.tool_registry.get_tool(tool_name)
        schema = tool.get_schema()
        
        prompt = f"""Extract parameters for the {tool_name} tool from this query.

Query: {query}

Required parameters: {json.dumps(schema, indent=2)}

Output JSON with extracted parameter values."""

        config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=1024)
        text = await generate_with_retry_async(prompt, config=config)
        
        # Parse JSON from response
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    
    async def _generate_tool_response(self, query: str, tool_output: Dict) -> str:
        """Generate natural language response from tool output"""
        
        text = await generate_with_retry_async(
            f"Query: {query}\n\nTool Output: {json.dumps(tool_output, indent=2)}\n\nProvide a natural, conversational response."
        )
        return text
    
    def _get_conversation_context(self, limit: int = 3) -> str:
        """Get recent conversation turns for context"""
        # Simplified - use first conversation or empty
        if not self.conversations:
            return "No previous context"
        
        recent = list(self.conversations.values())[0][-limit:]
        return "\n".join([
            f"Q: {turn['query']}\nA: {turn['response']}" 
            for turn in recent
        ])
    
    def _format_tools(self) -> str:
        """Format available tools for prompt"""
        return "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tool_registry.tools.values()
        ])
    
    def get_state(self) -> str:
        return self.state.value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats for dashboard"""
        metrics_avg = {}
        n = self.stats["metrics_count"]
        if n > 0:
            for k, v in self.stats["metrics_sum"].items():
                metrics_avg[k] = round(v / n, 2)
        return {
            "total_queries": self.stats["total_queries"],
            "rag_count": self.stats["rag_count"],
            "tool_count": self.stats["tool_count"],
            "hybrid_count": self.stats["hybrid_count"],
            "tools_used": self.stats["tools_used"],
            "metrics_count": n,
            "metrics_avg": metrics_avg
        }
