import * as React from "react";
import { SearchBar } from "@/components/SearchBar";
import { AnswerPanel } from "@/components/AnswerPanel";
import { SourcesList } from "@/components/SourcesList";
import { EvidenceDrawer } from "@/components/EvidenceDrawer";
import { ThemeToggle } from "@/components/ThemeToggle";
import { streamAsk } from "@/lib/api";
import type { Candidate, Source } from "@/lib/types";

type Phase = "idle" | "retrieving" | "streaming" | "done" | "error";

interface QueryState {
  phase: Phase;
  question: string;
  sources: Source[];
  candidates: Candidate[];
  answerText: string;
  modelUsed: string;
  contextUsed: boolean | null;
  error: string | null;
}

const initialState: QueryState = {
  phase: "idle",
  question: "",
  sources: [],
  candidates: [],
  answerText: "",
  modelUsed: "",
  contextUsed: null,
  error: null,
};

export default function App() {
  const [state, setState] = React.useState<QueryState>(initialState);
  const abortRef = React.useRef<AbortController | null>(null);

  const runQuery = React.useCallback(async (question: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState({
      ...initialState,
      phase: "retrieving",
      question,
    });

    try {
      for await (const evt of streamAsk(question, controller.signal)) {
        if (controller.signal.aborted) break;

        if (evt.event === "context") {
          setState((s) => ({
            ...s,
            phase: "streaming",
            sources: evt.data.sources,
            candidates: evt.data.candidates,
          }));
        } else if (evt.event === "token") {
          setState((s) => ({ ...s, answerText: s.answerText + evt.data.text }));
        } else if (evt.event === "done") {
          setState((s) => ({
            ...s,
            phase: "done",
            modelUsed: evt.data.model_used,
            contextUsed: evt.data.context_used,
          }));
        } else if (evt.event === "error") {
          setState((s) => ({ ...s, phase: "error", error: evt.data.message }));
        }
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      setState((s) => ({
        ...s,
        phase: "error",
        error: err instanceof Error ? err.message : String(err),
      }));
    }
  }, []);

  const handleCancel = React.useCallback(() => {
    abortRef.current?.abort();
    setState((s) => ({ ...s, phase: "done" }));
  }, []);

  const busy = state.phase === "retrieving" || state.phase === "streaming";

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <div className="flex items-baseline gap-2">
            <h1 className="text-lg font-semibold tracking-tight">arxiv-rag</h1>
            <span className="text-xs text-muted-foreground">
              retrieval-augmented answers over arXiv
            </span>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-6 py-8">
        <div className="mx-auto max-w-3xl space-y-8">
          <SearchBar onSubmit={runQuery} busy={busy} onCancel={handleCancel} />

          {state.question && (
            <div className="text-sm text-muted-foreground">
              <span className="font-medium text-foreground">Q:</span> {state.question}
            </div>
          )}

          <AnswerPanel
            phase={state.phase}
            answerText={state.answerText}
            modelUsed={state.modelUsed}
            contextUsed={state.contextUsed}
            error={state.error}
          />

          <SourcesList sources={state.sources} loading={state.phase === "retrieving"} />

          <EvidenceDrawer candidates={state.candidates} />
        </div>
      </main>

      <footer className="mx-auto max-w-4xl px-6 py-6 text-center text-xs text-muted-foreground">
        Free-tier LLM via OpenRouter. Embeddings + rerank run locally.
      </footer>
    </div>
  );
}
