import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Sparkles } from "lucide-react";

type Phase = "idle" | "retrieving" | "streaming" | "done" | "error";

interface AnswerPanelProps {
  phase: Phase;
  answerText: string;
  modelUsed: string;
  contextUsed: boolean | null;
  error: string | null;
}

/**
 * The main output card. Shows a skeleton while retrieval is running, then the
 * live-streaming answer with a blinking caret, then a static "done" state.
 */
export function AnswerPanel({
  phase,
  answerText,
  modelUsed,
  contextUsed,
  error,
}: AnswerPanelProps) {
  if (phase === "idle") return null;

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Sparkles className="h-4 w-4 text-muted-foreground" />
          Answer
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        {phase === "retrieving" ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-11/12" />
            <Skeleton className="h-4 w-4/5" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        ) : phase === "error" ? (
          <p className="text-sm text-destructive whitespace-pre-wrap">
            {error ?? "Something went wrong."}
          </p>
        ) : contextUsed === false && phase === "done" ? (
          <p className="text-sm text-muted-foreground">
            No indexed material matched this question. Try running an ingest
            first, or rephrase.
          </p>
        ) : (
          <div className="prose prose-sm max-w-none leading-relaxed text-foreground">
            <p className="whitespace-pre-wrap">
              {answerText}
              {phase === "streaming" && <span className="cursor-blink" aria-hidden />}
            </p>
          </div>
        )}
      </CardContent>
      {phase === "done" && modelUsed && (
        <div className="border-t border-border px-6 py-3 text-xs text-muted-foreground">
          Answered by <code className="font-mono">{modelUsed}</code>
        </div>
      )}
    </Card>
  );
}
