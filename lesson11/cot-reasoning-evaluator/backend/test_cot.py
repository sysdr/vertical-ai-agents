"""Tests for CoT reasoning system"""
import pytest
from cot_agent import CoTPromptBuilder, ReasoningTraceParser, QualityEvaluator

def test_prompt_building():
    """Test CoT prompt construction"""
    query = "What is 5 + 3?"
    prompt = CoTPromptBuilder.build_cot_prompt(query)
    assert "step-by-step" in prompt.lower()
    assert query in prompt

def test_step_extraction():
    """Test parsing reasoning steps"""
    trace = """1. First, I identify that we have 5 apples.
2. Then, we add 3 more apples.
3. Therefore, 5 + 3 = 8 apples."""
    
    steps = ReasoningTraceParser.extract_steps(trace)
    assert len(steps) == 3
    assert "5 apples" in steps[0]

def test_clarity_scoring():
    """Test clarity assessment"""
    good_steps = [
        "First, we identify the initial count of 5 items",
        "Then, we add 3 more items to the collection",
        "Therefore, the final count is 8 items"
    ]
    
    score = QualityEvaluator.assess_clarity(good_steps)
    assert score > 0.5

def test_conclusion_detection():
    """Test conclusion extraction"""
    trace = "Step 1: Analyze. Step 2: Calculate. Therefore, the answer is 42."
    conclusion = ReasoningTraceParser.extract_conclusion(trace)
    assert conclusion is not None
    assert "42" in conclusion

def test_overall_evaluation():
    """Test comprehensive quality scoring"""
    trace = """1. First, identify that John has 3 apples.
2. Then, John buys 2 more apples, so 3 + 2 = 5 apples.
3. Next, John gives 1 apple to Mary, so 5 - 1 = 4 apples.
4. Therefore, John has 4 apples remaining."""
    
    scores = QualityEvaluator.evaluate_reasoning(trace)
    assert scores["step_count"] >= 3
    assert scores["overall_quality"] > 0.6
    assert scores["conclusion_present"] == 1.0
