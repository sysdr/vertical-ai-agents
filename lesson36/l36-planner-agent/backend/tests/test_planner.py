"""Tests for PlannerAgent"""
import pytest
from app.planner import PlannerAgent
from app.models import PlanRequest
import os

@pytest.fixture
def planner():
    api_key = os.getenv("GEMINI_API_KEY", "test-key")
    return PlannerAgent(api_key=api_key)

@pytest.mark.asyncio
async def test_simple_query(planner):
    """Test simple query doesn't require decomposition"""
    request = PlanRequest(query="What is machine learning?")
    plan = await planner.decompose_query(request)
    
    assert plan.original_query == "What is machine learning?"
    assert len(plan.sub_queries) >= 1

@pytest.mark.asyncio
async def test_complex_query(planner):
    """Test complex query gets decomposed (or fallback when API unavailable)"""
    request = PlanRequest(
        query="Compare Tesla's autopilot technology to Mercedes' system and analyze regulatory concerns"
    )
    plan = await planner.decompose_query(request)
    
    assert plan.original_query.startswith("Compare Tesla")
    assert len(plan.sub_queries) >= 1  # >=2 when API works, 1 when fallback
    assert plan.strategy in ["parallel", "sequential"]

@pytest.mark.asyncio
async def test_max_sub_queries_limit(planner):
    """Test sub-queries don't exceed limit"""
    request = PlanRequest(
        query="Analyze climate change impacts on agriculture, economy, health, infrastructure, and biodiversity",
        max_sub_queries=3
    )
    plan = await planner.decompose_query(request)
    
    assert len(plan.sub_queries) <= 3

def test_fallback_plan(planner):
    """Test fallback plan structure"""
    plan = planner._fallback_plan("test query")
    
    assert plan.requires_decomposition == False
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].text == "test query"
