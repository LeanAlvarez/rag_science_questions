import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ExternalLink, BookOpen } from "lucide-react";
import type { Source } from "@/lib/types";

interface SourcesListProps {
  sources: Source[];
  loading: boolean;
}

/**
 * The papers cited by (or, before the LLM finishes, that WILL be cited by)
 * the answer. Rendered as soon as the "context" SSE event fires — the user
 * gets to see them before the text starts streaming.
 */
export function SourcesList({ sources, loading }: SourcesListProps) {
  if (loading) {
    return (
      <section className="w-full space-y-3">
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Sources
        </h2>
        <div className="grid gap-2">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-14 w-full" />
          ))}
        </div>
      </section>
    );
  }

  if (sources.length === 0) return null;

  return (
    <section className="w-full space-y-3">
      <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
        Sources ({sources.length})
      </h2>
      <div className="grid gap-2">
        {sources.map((src, i) => (
          <Card key={src.arxiv_id} className="transition-colors hover:bg-accent">
            <CardContent className="flex items-start gap-3 p-4">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
                <span className="text-xs font-semibold">[{i + 1}]</span>
              </div>
              <div className="min-w-0 flex-1">
                <a
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex items-start gap-2"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-foreground group-hover:underline">
                      {src.title}
                    </p>
                    <p className="mt-0.5 font-mono text-xs text-muted-foreground">
                      arxiv:{src.arxiv_id}
                    </p>
                  </div>
                  <ExternalLink className="h-3.5 w-3.5 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
                </a>
              </div>
              <BookOpen className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
