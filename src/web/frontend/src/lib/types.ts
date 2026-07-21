// Wire types — must stay in sync with src/web/api.py Pydantic models and
// src/query/pipeline.py's answer_question_stream() event shapes.

export interface Source {
  arxiv_id: string;
  title: string;
  url: string;
}

export interface Candidate {
  chunk_id: number;
  arxiv_id: string;
  chunk_index: number;
  content: string;
  title: string;
  vector_similarity: number | null;
  vector_rank: number | null;
  keyword_score: number | null;
  keyword_rank: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
}

// SSE event payloads emitted by /api/ask/stream.
export type StreamEvent =
  | { event: "context"; data: { sources: Source[]; candidates: Candidate[] } }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: { model_used: string; context_used: boolean } }
  | { event: "error"; data: { message: string } };

// The full non-streaming response from /api/ask.
export interface AnswerResponse {
  text: string;
  sources: Source[];
  candidates: Candidate[];
  model_used: string;
  context_used: boolean;
}
