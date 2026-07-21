import * as React from "react";
import { Search, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface SearchBarProps {
  onSubmit: (question: string) => void;
  busy: boolean;
  onCancel?: () => void;
}

/**
 * The one input at the top of the page. Autofocused, submit-on-Enter, and
 * turns into a "Stop" button while a stream is in flight.
 */
export function SearchBar({ onSubmit, busy, onCancel }: SearchBarProps) {
  const [value, setValue] = React.useState("");
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = value.trim();
    if (!q || busy) return;
    onSubmit(q);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex w-full items-center gap-2"
    >
      <div className="relative flex-1">
        <Search
          className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden
        />
        <Input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask a question about the indexed arXiv papers…"
          className="h-12 pl-10 text-base"
          disabled={busy}
          aria-label="Question"
        />
      </div>
      {busy && onCancel ? (
        <Button type="button" variant="outline" onClick={onCancel} className="h-12 px-6">
          <Loader2 className="h-4 w-4 animate-spin" />
          Stop
        </Button>
      ) : (
        <Button type="submit" disabled={!value.trim() || busy} className="h-12 px-6">
          Ask
        </Button>
      )}
    </form>
  );
}
