import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, GitBranch, Wrench, Brain, Activity, ShieldCheck, Layers3 } from "lucide-react";

function Node({
  title,
  subtitle,
  icon,
  className = "",
}: {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-2xl border bg-white shadow-sm p-4 min-h-[92px] flex flex-col justify-center ${className}`}>
      <div className="flex items-center gap-2 font-semibold text-slate-900">
        {icon}
        <span>{title}</span>
      </div>
      {subtitle ? <div className="text-sm text-slate-600 mt-1">{subtitle}</div> : null}
    </div>
  );
}

function ArrowRow({ label }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 text-slate-500 py-2">
      <ArrowRight className="w-4 h-4" />
      {label ? <span className="text-xs uppercase tracking-wide">{label}</span> : null}
    </div>
  );
}

function MiniCallout({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-2xl border bg-slate-50 p-4">
      <div className="text-sm font-semibold text-slate-900 mb-2">{title}</div>
      <div className="space-y-2">
        {items.map((item) => (
          <div key={item} className="text-sm text-slate-700 flex gap-2">
            <span className="mt-[7px] h-1.5 w-1.5 rounded-full bg-slate-400 shrink-0" />
            <span>{item}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ArchitectureComparisonDiagrams() {
  return (
    <div className="min-h-screen bg-slate-100 p-6 md:p-10">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <Badge className="rounded-full px-3 py-1 text-sm">Architecture Comparison</Badge>
            <Badge variant="secondary" className="rounded-full px-3 py-1 text-sm">
              Meeting Visual
            </Badge>
          </div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-slate-900">
            Sub-Agent Route vs. Your Newer Project
          </h1>
          <p className="text-slate-600 max-w-3xl text-base md:text-lg">
            Both designs are agentic, but they place complexity in different places.
            The left side delegates to specialized child agents. The right side keeps control centralized in one orchestrator,
            using skills to scope tools and behavior.
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <Card className="rounded-3xl shadow-lg border-0 bg-white">
            <CardHeader className="pb-2">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-10 h-10 rounded-2xl bg-slate-900 text-white flex items-center justify-center">
                  <GitBranch className="w-5 h-5" />
                </div>
                <div>
                  <CardTitle className="text-2xl">Sub-Agent Architecture</CardTitle>
                  <div className="text-sm text-slate-600 mt-1">Delegation-first</div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <Node title="User Request" subtitle="One request enters the system" icon={<span>👤</span>} />
              <ArrowRow label="delegate" />
              <Node
                title="Main Orchestrator"
                subtitle="Decides which specialist should take over"
                icon={<Brain className="w-4 h-4" />}
              />
              <ArrowRow label="handoff" />
              <div className="grid md:grid-cols-3 gap-4">
                <Node
                  title="Math Agent"
                  subtitle="Own prompt, own loop"
                  icon={<Layers3 className="w-4 h-4" />}
                  className="bg-slate-50"
                />
                <Node
                  title="Data Agent"
                  subtitle="Own prompt, own loop"
                  icon={<Layers3 className="w-4 h-4" />}
                  className="bg-slate-50"
                />
                <Node
                  title="Research Agent"
                  subtitle="Own prompt, own loop"
                  icon={<Layers3 className="w-4 h-4" />}
                  className="bg-slate-50"
                />
              </div>
              <ArrowRow label="tool use inside each agent" />
              <div className="grid md:grid-cols-3 gap-4">
                <Node title="Math Tools" subtitle="Calculator APIs, formulas" icon={<Wrench className="w-4 h-4" />} />
                <Node title="Data Tools" subtitle="SQL, files, analysis" icon={<Wrench className="w-4 h-4" />} />
                <Node title="Research Tools" subtitle="Search, retrieval, docs" icon={<Wrench className="w-4 h-4" />} />
              </div>

              <div className="grid md:grid-cols-2 gap-4 pt-2">
                <MiniCallout
                  title="What this optimizes for"
                  items={[
                    "Autonomous specialist workflows",
                    "Independent reasoning loops by domain",
                    "Strong domain separation",
                  ]}
                />
                <MiniCallout
                  title="What this costs"
                  items={[
                    "More handoffs between agents",
                    "More prompts and LLM calls",
                    "Harder tracing across boundaries",
                  ]}
                />
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-3xl shadow-lg border-0 bg-white">
            <CardHeader className="pb-2">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-10 h-10 rounded-2xl bg-slate-900 text-white flex items-center justify-center">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <div>
                  <CardTitle className="text-2xl">Your Newer Project</CardTitle>
                  <div className="text-sm text-slate-600 mt-1">Control-first</div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <Node title="User Request" subtitle="One request enters the host app" icon={<span>👤</span>} />
              <ArrowRow label="centralize" />
              <Node
                title="Main Orchestrator"
                subtitle="Single reasoning loop manages the turn"
                icon={<Brain className="w-4 h-4" />}
              />
              <ArrowRow label="scope behavior" />
              <Node
                title="Skill Selection"
                subtitle="Choose instructions + allowed tools for this turn"
                icon={<Layers3 className="w-4 h-4" />}
                className="bg-slate-50"
              />
              <ArrowRow label="bind tools" />
              <Node
                title="Unified MCP Runtime"
                subtitle="Shared tool runtime across MCP servers"
                icon={<Wrench className="w-4 h-4" />}
              />
              <ArrowRow label="call capabilities" />
              <div className="grid md:grid-cols-3 gap-4">
                <Node title="Calc Server" subtitle="MCP tools" icon={<Wrench className="w-4 h-4" />} />
                <Node title="Quote Server" subtitle="MCP tools" icon={<Wrench className="w-4 h-4" />} />
                <Node title="Data Tools" subtitle="MCP tools" icon={<Wrench className="w-4 h-4" />} />
              </div>
              <ArrowRow label="measure centrally" />
              <Node
                title="Sessions + Observability"
                subtitle="Metrics, traces, benchmarking, session state"
                icon={<Activity className="w-4 h-4" />}
              />

              <div className="grid md:grid-cols-2 gap-4 pt-2">
                <MiniCallout
                  title="What this optimizes for"
                  items={[
                    "Centralized control and routing",
                    "Cleaner observability and benchmarking",
                    "Fewer orchestration layers",
                  ]}
                />
                <MiniCallout
                  title="What this costs"
                  items={[
                    "Main orchestrator carries more responsibility",
                    "Less autonomous domain behavior by default",
                    "True sub-agents added only when justified",
                  ]}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="rounded-3xl border-0 shadow-lg bg-white">
          <CardContent className="p-6 md:p-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <MiniCallout
                title="Decision rule"
                items={[
                  "Use sub-agents when each domain needs its own independent reasoning loop.",
                  "Use your newer project when you want centralized orchestration, simpler control flow, and stronger observability.",
                ]}
              />
              <MiniCallout
                title="Meeting takeaway"
                items={[
                  "You are not avoiding specialization.",
                  "You are specializing through skills and tool scoping instead of through many autonomous child agents.",
                ]}
              />
              <MiniCallout
                title="Best one-line summary"
                items={[
                  "Sub-agent model: delegation-first.",
                  "Your newer project: control-first.",
                ]}
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
