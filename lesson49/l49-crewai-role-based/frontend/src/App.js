import React, { useState, useRef, useCallback } from "react";

const BACKEND = process.env.REACT_APP_BACKEND_URL || "http://localhost:8049";

const AGENTS = [
  { step: 1, role: "Senior Technical Researcher", icon: "🔬", color: "#e8f5e9", border: "#66bb6a" },
  { step: 2, role: "Principal Solution Architect", icon: "🏗️", color: "#fff3e0", border: "#ffa726" },
  { step: 3, role: "Technical Content Strategist", icon: "✍️", color: "#e3f2fd", border: "#42a5f5" },
];

function AgentCard({ agent, status, output }) {
  const statusColors = { idle: "#9e9e9e", active: "#ff9800", done: "#4caf50" };
  const statusLabels = { idle: "Waiting", active: "Working…", done: "Complete" };
  return (
    <div style={{
      background: status === "idle" ? "#fafafa" : agent.color,
      border: `2px solid ${status === "idle" ? "#e0e0e0" : agent.border}`,
      borderRadius: 12, padding: "18px 20px", marginBottom: 12,
      transition: "all 0.4s ease", boxShadow: status === "active" ? "0 4px 16px rgba(0,0,0,0.12)" : "0 1px 4px rgba(0,0,0,0.06)"
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <span style={{ fontSize: 24 }}>{agent.icon}</span>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, color: "#1a1a1a" }}>{agent.role}</div>
          <div style={{ fontSize: 12, color: statusColors[status], fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
            {status === "active" && <span style={{ marginRight: 6 }}>⚡</span>}
            {statusLabels[status]}
          </div>
        </div>
        <div style={{ marginLeft: "auto", fontSize: 12, color: "#9e9e9e" }}>Step {agent.step}</div>
      </div>
      {output && (
        <div style={{
          background: "rgba(255,255,255,0.7)", borderRadius: 8, padding: "10px 12px",
          fontSize: 13, color: "#333", lineHeight: 1.6, whiteSpace: "pre-wrap",
          maxHeight: 180, overflowY: "auto", marginTop: 6,
          fontFamily: "'Courier New', monospace",
        }}>
          {output}
        </div>
      )}
    </div>
  );
}

function PipelineFlow({ activeStep }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, margin: "20px 0 24px", flexWrap: "wrap" }}>
      {AGENTS.map((agent, i) => (
        <React.Fragment key={agent.step}>
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
            opacity: activeStep < agent.step ? 0.4 : 1, transition: "opacity 0.3s"
          }}>
            <div style={{
              width: 44, height: 44, borderRadius: "50%", display: "flex", alignItems: "center",
              justifyContent: "center", fontSize: 20,
              background: activeStep >= agent.step ? agent.color : "#f5f5f5",
              border: `2px solid ${activeStep >= agent.step ? agent.border : "#ddd"}`,
              boxShadow: activeStep === agent.step ? `0 0 12px ${agent.border}88` : "none",
            }}>{agent.icon}</div>
            <div style={{ fontSize: 10, color: "#666", textAlign: "center", maxWidth: 70, lineHeight: 1.3 }}>
              {agent.role.split(" ").slice(-1)[0]}
            </div>
          </div>
          {i < AGENTS.length - 1 && (
            <div style={{
              width: 40, height: 2, background: activeStep > agent.step ? "#66bb6a" : "#e0e0e0",
              transition: "background 0.5s", marginTop: -12,
            }} />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

export default function App() {
  const [topic, setTopic] = useState("");
  const [running, setRunning] = useState(false);
  const [agentStatuses, setAgentStatuses] = useState({ 1: "idle", 2: "idle", 3: "idle" });
  const [agentOutputs, setAgentOutputs] = useState({ 1: "", 2: "", 3: "" });
  const [finalOutput, setFinalOutput] = useState("");
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState("");
  const [runTime, setRunTime] = useState(null);
  const startTimeRef = useRef(null);
  const esRef = useRef(null);
  const completedAgents = Object.values(agentStatuses).filter(v => v === "done").length;
  const completedTasks = Object.values(agentOutputs).filter(v => (v || "").trim().length > 0).length;
  const outputChars = finalOutput.length;

  const reset = () => {
    setAgentStatuses({ 1: "idle", 2: "idle", 3: "idle" });
    setAgentOutputs({ 1: "", 2: "", 3: "" });
    setFinalOutput("");
    setActiveStep(0);
    setError("");
    setRunTime(null);
  };

  const runCrew = useCallback(() => {
    if (!topic.trim() || running) return;
    reset();
    setRunning(true);
    startTimeRef.current = Date.now();

    const url = `${BACKEND}/stream-crew?topic=${encodeURIComponent(topic.trim())}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);

      if (data.type === "agent_start") {
        const step = data.step;
        setActiveStep(step);
        setAgentStatuses(prev => {
          const next = { ...prev };
          if (step > 1) next[step - 1] = "done";
          next[step] = "active";
          return next;
        });
      }

      if (data.type === "task_complete") {
        setAgentOutputs(prev => ({ ...prev, [data.step]: data.output }));
      }

      if (data.type === "crew_complete") {
        setAgentStatuses({ 1: "done", 2: "done", 3: "done" });
        setActiveStep(4);
        setFinalOutput(data.final_output);
        setRunTime(((Date.now() - startTimeRef.current) / 1000).toFixed(1));
        setRunning(false);
        es.close();
      }
    };

    es.onerror = () => {
      setError("Connection error. Check backend is running on port 8049.");
      setRunning(false);
      es.close();
    };
  }, [topic, running]);

  const exampleTopics = [
    "Speculative decoding in LLM inference",
    "Kubernetes VPA vs HPA for ML workloads",
    "RAG vs fine-tuning for domain adaptation",
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#f8f9fa", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
      <div style={{ maxWidth: 760, margin: "0 auto", padding: "40px 20px" }}>

        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 36 }}>
          <div style={{ display: "inline-block", background: "#1a1a2e", color: "#fff", borderRadius: 8, padding: "4px 14px", fontSize: 12, fontWeight: 700, letterSpacing: 1, marginBottom: 12 }}>
            L49 · MODULE 5 · CREWAI
          </div>
          <h1 style={{ margin: "8px 0 6px", fontSize: 28, fontWeight: 800, color: "#1a1a1a" }}>
            Role-Based Execution
          </h1>
          <p style={{ color: "#666", fontSize: 15, margin: 0 }}>
            3-agent sequential crew · Gemini 2.0 Flash · Researcher → Analyst → Writer
          </p>
        </div>

        {/* Input */}
        <div style={{ background: "#fff", borderRadius: 16, padding: 24, boxShadow: "0 2px 12px rgba(0,0,0,0.07)", marginBottom: 24 }}>
          <label style={{ fontWeight: 600, fontSize: 14, color: "#444", display: "block", marginBottom: 8 }}>
            Research Topic
          </label>
          <div style={{ display: "flex", gap: 10 }}>
            <input
              value={topic}
              onChange={e => setTopic(e.target.value)}
              onKeyDown={e => e.key === "Enter" && runCrew()}
              placeholder="e.g. Speculative decoding in LLM inference"
              disabled={running}
              style={{
                flex: 1, padding: "12px 16px", borderRadius: 10, border: "2px solid #e0e0e0",
                fontSize: 15, outline: "none", transition: "border 0.2s",
                borderColor: topic ? "#42a5f5" : "#e0e0e0",
              }}
            />
            <button
              onClick={runCrew}
              disabled={!topic.trim() || running}
              style={{
                padding: "12px 24px", borderRadius: 10, border: "none", cursor: running ? "not-allowed" : "pointer",
                background: !topic.trim() || running ? "#e0e0e0" : "linear-gradient(135deg, #1a237e, #1565c0)",
                color: !topic.trim() || running ? "#aaa" : "#fff",
                fontWeight: 700, fontSize: 15, transition: "all 0.2s",
                minWidth: 110,
              }}
            >
              {running ? "⟳ Running" : "▶ Run Crew"}
            </button>
          </div>

          {/* Example topics */}
          <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
            {exampleTopics.map(t => (
              <button key={t} onClick={() => setTopic(t)} disabled={running}
                style={{
                  padding: "4px 12px", borderRadius: 20, border: "1px solid #e0e0e0",
                  background: "#fafafa", cursor: "pointer", fontSize: 12, color: "#555",
                  transition: "all 0.2s",
                }}>
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Pipeline visualization */}
        <PipelineFlow activeStep={activeStep} />

        {/* Live metrics */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10, marginBottom: 14 }}>
          <div style={{ background: "#fff", borderRadius: 10, padding: 12, boxShadow: "0 1px 6px rgba(0,0,0,0.06)" }}>
            <div style={{ fontSize: 11, color: "#777", textTransform: "uppercase", letterSpacing: 0.4 }}>Agents Complete</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: "#1a1a1a" }}>{completedAgents}</div>
          </div>
          <div style={{ background: "#fff", borderRadius: 10, padding: 12, boxShadow: "0 1px 6px rgba(0,0,0,0.06)" }}>
            <div style={{ fontSize: 11, color: "#777", textTransform: "uppercase", letterSpacing: 0.4 }}>Tasks With Output</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: "#1a1a1a" }}>{completedTasks}</div>
          </div>
          <div style={{ background: "#fff", borderRadius: 10, padding: 12, boxShadow: "0 1px 6px rgba(0,0,0,0.06)" }}>
            <div style={{ fontSize: 11, color: "#777", textTransform: "uppercase", letterSpacing: 0.4 }}>Output Characters</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: "#1a1a1a" }}>{outputChars}</div>
          </div>
          <div style={{ background: "#fff", borderRadius: 10, padding: 12, boxShadow: "0 1px 6px rgba(0,0,0,0.06)" }}>
            <div style={{ fontSize: 11, color: "#777", textTransform: "uppercase", letterSpacing: 0.4 }}>Runtime (s)</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: "#1a1a1a" }}>{runTime || 0}</div>
          </div>
        </div>

        {/* Agent cards */}
        {AGENTS.map(agent => (
          <AgentCard
            key={agent.step}
            agent={agent}
            status={agentStatuses[agent.step]}
            output={agentOutputs[agent.step]}
          />
        ))}

        {/* Error */}
        {error && (
          <div style={{ background: "#ffebee", border: "1px solid #ef9a9a", borderRadius: 10, padding: 14, color: "#c62828", fontSize: 14, marginTop: 12 }}>
            ⚠ {error}
          </div>
        )}

        {/* Final output */}
        {finalOutput && (
          <div style={{ background: "#fff", borderRadius: 16, padding: 24, boxShadow: "0 2px 12px rgba(0,0,0,0.07)", marginTop: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "#1a1a1a" }}>✅ Technical Briefing</h3>
              {runTime && <span style={{ fontSize: 12, color: "#9e9e9e", background: "#f5f5f5", padding: "3px 10px", borderRadius: 20 }}>{runTime}s · 3 agents · 3 tasks</span>}
            </div>
            <div style={{
              background: "#f8f9fa", borderRadius: 10, padding: "16px 18px",
              fontSize: 14, lineHeight: 1.8, color: "#333", whiteSpace: "pre-wrap",
              maxHeight: 500, overflowY: "auto",
            }}>
              {finalOutput}
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ textAlign: "center", marginTop: 40, color: "#bbb", fontSize: 12 }}>
          VAIA Curriculum · L49 of 90 · Module 5: Multi-Agent Architectures
        </div>
      </div>
    </div>
  );
}
