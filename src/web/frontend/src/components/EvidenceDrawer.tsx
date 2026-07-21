import * as React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { Candidate } from "@/lib/types";

interface EvidenceDrawerProps {
  candidates: Candidate[];
}

/**
 * Collapsible evidence panel. When open, shows every reranked fragment with
 * ALL of its scores (vector, keyword, RRF, cross-encoder). This is the payoff
 * of keeping the scores side-by-side in Candidate — the user (or you, when
 * debugging) can see WHY each fragment was surfaced.
 */
export function EvidenceDrawer({ candidates }: EvidenceDrawerProps) {
  const [open, setOpen] = React.useState(false);

  if (candidates.length === 0) return null;

  return (
    <section className="w-full space-y-3">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setOpen((v) => !v)}
        className="h-8 gap-2 px-2 text-xs font-medium uppercase tracking-wider text-muted-foreground hover:text-foreground"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
        Evidence ({candidates.length} fragments)
      </Button>

      {open && (
        <div className="space-y-2">
          {candidates.map((c, i) => (
            <Card key={c.chunk_id}>
              <CardContent className="space-y-2 p-4">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                  <span className="inline-flex items-center rounded bg-muted px-1.5 py-0.5 font-semibold">
                    [{i + 1}]
                  </span>
                  <span className="font-mono text-muted-foreground">
                    arxiv:{c.arxiv_id}
                  </span>
                  <span className="text-muted-foreground">chunk #{c.chunk_index}</span>
                </div>

                <p className="text-sm font-medium text-foreground">{c.title}</p>

                <ScoresGrid c={c} />

                <p className="text-xs leading-relaxed text-muted-foreground line-clamp-4">
                  {c.content}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </section>
  );
}

function ScoresGrid({ c }: { c: Candidate }) {
  const items: Array<[string, number | null, string]> = [
    ["vec_sim", c.vector_similarity, ".3f"],
    ["kw", c.keyword_score, ".3f"],
    ["rrf", c.rrf_score, ".4f"],
    ["rerank", c.rerank_score, "+.3f"],
  ];
  return (
    <div className="grid grid-cols-4 gap-2 rounded-md border border-border bg-background/50 p-2">
      {items.map(([label, val, fmt]) => (
        <div key={label} className="flex flex-col items-center text-center">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
            {label}
          </span>
          <span className="font-mono text-xs text-foreground">{fmtNum(val, fmt)}</span>
        </div>
      ))}
    </div>
  );
}

function fmtNum(n: number | null, spec: string): string {
  if (n === null || n === undefined) return "—";
  // Minimal format-spec support: '.3f', '.4f', '+.3f'
  const wantSign = spec.startsWith("+");
  const digitsMatch = spec.match(/\.(\d+)f$/);
  const digits = digitsMatch ? Number(digitsMatch[1]) : 3;
  const rounded = n.toFixed(digits);
  return wantSign && n >= 0 ? `+${rounded}` : rounded;
}
