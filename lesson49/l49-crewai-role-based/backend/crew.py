"""
L49 Core: CrewAI Role-Based Crew
Three-agent sequential pipeline: Researcher → Analyst → Writer
"""
import os
from crewai import Agent, Task, Crew, Process, LLM

def build_llm(temperature: float = 0.3):
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature,
    )

def build_crew(topic: str, verbose: bool = False):
    llm = build_llm()

    # ── Agents ───────────────────────────────────────────────────────────────
    researcher = Agent(
        role="Senior Technical Researcher",
        goal=f"Surface non-obvious technical insights about: {topic}",
        backstory=(
            "You are a PhD-level systems researcher with 10 years at a top AI lab. "
            "You distrust surface-level summaries and always identify precise mechanisms, "
            "edge cases, and production gotchas that practitioners overlook."
        ),
        llm=llm,
        verbose=verbose,
        max_iter=3,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Principal Solution Architect",
        goal=f"Identify enterprise deployment patterns and trade-offs for: {topic}",
        backstory=(
            "You are a Principal Architect who has shipped 40+ production AI systems at "
            "companies like Stripe, Uber, and Netflix. You translate raw research into "
            "ranked, actionable architectural decisions with explicit cost/risk trade-offs."
        ),
        llm=llm,
        verbose=verbose,
        max_iter=3,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Content Strategist",
        goal=f"Produce a precise, engineer-targeted briefing on: {topic}",
        backstory=(
            "You write for Staff+ engineers. You assume deep technical fluency "
            "and never over-explain basics. Your briefings are structured, "
            "scannable, and always close with a concrete 'what to do Monday morning'."
        ),
        llm=build_llm(temperature=0.5),
        verbose=verbose,
        max_iter=3,
        allow_delegation=False,
    )

    # ── Tasks ────────────────────────────────────────────────────────────────
    research_task = Task(
        description=(
            f"Research the topic: '{topic}'. "
            "Identify 5 concrete technical insights that would surprise a senior engineer. "
            "For each insight include: (1) the specific claim, (2) the underlying mechanism, "
            "(3) the production implication. Do NOT include general overviews."
        ),
        expected_output=(
            "Numbered list of exactly 5 insights. Each formatted as:\n"
            "N. [CLAIM] | [MECHANISM] | [PRODUCTION IMPLICATION]"
        ),
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "Using the research findings above, produce a ranked architectural analysis. "
            "Identify the top 3 deployment patterns, each with a clear trade-off matrix "
            "(latency vs cost vs complexity). Flag at least 1 common enterprise anti-pattern."
        ),
        expected_output=(
            "Structured analysis with:\n"
            "- 3 deployment patterns (name | trade-offs | when to use)\n"
            "- 1 anti-pattern (name | why teams fall into it | mitigation)"
        ),
        agent=analyst,
        context=[research_task],
    )

    writing_task = Task(
        description=(
            "Using the research and architectural analysis above, write a technical briefing "
            f"for Staff+ engineers on '{topic}'. Include: TL;DR (2 sentences), "
            "key insights (bullet list), architectural recommendations, and a 'Monday Action' "
            "section with 1 specific thing an engineer can implement this week."
        ),
        expected_output=(
            "A polished technical briefing with sections: "
            "TL;DR | Key Insights | Architecture Recommendations | Monday Action"
        ),
        agent=writer,
        context=[research_task, analysis_task],
    )

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=verbose,
    )

def run_crew(topic: str) -> dict:
    """Execute crew and return structured results."""
    try:
        crew = build_crew(topic, verbose=True)
        result = crew.kickoff(inputs={"topic": topic})

        task_outputs = []
        for i, task in enumerate(crew.tasks):
            task_outputs.append({
                "step": i + 1,
                "agent": crew.agents[i].role,
                "output": task.output.raw if hasattr(task, "output") and task.output else "",
            })

        return {
            "topic": topic,
            "final_output": str(result),
            "task_outputs": task_outputs,
            "agent_count": len(crew.agents),
            "task_count": len(crew.tasks),
        }
    except Exception as e:
        # Keep the lesson demo functional even if external LLM credentials are invalid.
        fallback_outputs = [
            {
                "step": 1,
                "agent": "Senior Technical Researcher",
                "output": f"[DEMO MODE] Key research findings for '{topic}': mechanism details, edge cases, and production implications.",
            },
            {
                "step": 2,
                "agent": "Principal Solution Architect",
                "output": "[DEMO MODE] Ranked deployment patterns with latency/cost/complexity trade-offs and one anti-pattern mitigation.",
            },
            {
                "step": 3,
                "agent": "Technical Content Strategist",
                "output": "[DEMO MODE] Staff+ briefing drafted with TL;DR, insights, architecture recommendations, and Monday action.",
            },
        ]
        final_output = (
            f"[DEMO MODE] Crew execution fell back due to provider error: {e}\n\n"
            "TL;DR: The pipeline still demonstrates a 3-agent sequential flow.\n"
            "Key Insights: Non-obvious mechanism behavior, deployment trade-offs, and implementation guidance.\n"
            "Architecture Recommendations: Start with observability-first rollout and staged traffic ramp.\n"
            "Monday Action: Instrument one critical path and compare baseline vs optimized behavior."
        )
        return {
            "topic": topic,
            "final_output": final_output,
            "task_outputs": fallback_outputs,
            "agent_count": 3,
            "task_count": 3,
        }
