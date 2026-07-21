import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges Tailwind class strings so later utilities override earlier ones
 * without producing duplicate/conflicting class output. Same helper shadcn
 * ships with — every UI component uses it via a `className` prop.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
