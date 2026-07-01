"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

type MarkdownMathProps = {
  content: string;
};

const CITATION_SENTINEL = "__AI_BRAIN_CITATION__";

const INLINE_CITATION_PATTERN =
  /\[([^\]\n]+?)(?:,\s*|\s+)p\.?\s*(\d+)(?::c\d+)?(?:\s+chunk\s+[^\]]+)?\]/gi;

function cleanCitationDocument(value: string): string {
  return value
    .trim()
    .replace(/^[0-9a-f]{64}_/i, "")
    .replace(/\.pdf$/i, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ");
}

function formatInlineCitations(content: string): string {
  return content.replace(
    INLINE_CITATION_PATTERN,
    (_match, rawDocument: string, page: string) => {
      const document = cleanCitationDocument(rawDocument);
      const label = `${document} · p.${page}`;

      return `\`${CITATION_SENTINEL}${label}\``;
    },
  );
}

export function MarkdownMath({ content }: MarkdownMathProps) {
  const formattedContent = formatInlineCitations(content);

  return (
    <div className="min-w-0 text-sm leading-7 text-slate-700">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        skipHtml
        components={{
          p: ({ children }) => (
            <p className="mb-3 last:mb-0">{children}</p>
          ),
          ul: ({ children }) => (
            <ul className="mb-3 list-disc space-y-1 pl-5">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-3 list-decimal space-y-1 pl-5">
              {children}
            </ol>
          ),
          li: ({ children }) => <li>{children}</li>,
          code: ({ children, className }) => {
            const isBlock = Boolean(className);
            const value = String(children).replace(/\n$/, "");

            if (
              !isBlock &&
              value.startsWith(CITATION_SENTINEL)
            ) {
              return (
                <span
                  className="mx-0.5 inline text-[11px] font-normal leading-5 text-slate-400"
                  aria-label="문서 출처"
                >
                  [{value.slice(CITATION_SENTINEL.length)}]
                </span>
              );
            }

            return isBlock ? (
              <pre className="mb-3 overflow-x-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-100">
                <code className={className}>{children}</code>
              </pre>
            ) : (
              <code className="rounded bg-slate-100 px-1 py-0.5 text-[0.9em]">
                {children}
              </code>
            );
          },
        }}
      >
        {formattedContent}
      </ReactMarkdown>
    </div>
  );
}
