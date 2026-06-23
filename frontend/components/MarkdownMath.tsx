"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

type MarkdownMathProps = {
  content: string;
};

export function MarkdownMath({ content }: MarkdownMathProps) {
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
            <ul className="mb-3 list-disc space-y-1 pl-5">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-3 list-decimal space-y-1 pl-5">{children}</ol>
          ),
          li: ({ children }) => <li>{children}</li>,
          code: ({ children, className }) => {
            const isBlock = Boolean(className);

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
        {content}
      </ReactMarkdown>
    </div>
  );
}
