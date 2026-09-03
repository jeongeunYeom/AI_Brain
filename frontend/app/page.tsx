"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  analyzeImage,
  askComparison,
  askQuestion,
  ChatCompareResponse,
  ChatResponse,
  createUploadJob,
  getUploadJob,
  uploadDocument,
  UploadJob,
} from "@/lib/api";
import { DocumentInfoPanel } from "@/components/DocumentInfoPanel";
import { PlotPanel } from "@/components/PlotPanel";
import { SystemStatusPanel } from "@/components/SystemStatusPanel";
import { MarkdownMath } from "@/components/MarkdownMath";
import { AppIconRail, MobileModeTabs } from "@/components/AppNavigation";


type FigureReference = {
  document: string;
  page?: number | null;
  title?: string | null;
  image_type?: string | null;
  filename: string;
  url: string;
  preview_url?: string | null;
};

type ChatResponseWithFigures = ChatResponse & {
  figures?: FigureReference[];
};

type ChatCompareResponseWithFigures = Omit<
  ChatCompareResponse,
  "figures"
> & {
  figures?: FigureReference[];
};

type AnswerMode =
  | "qwen3:8b"
  | "gemma4:latest"
  | "compare";


const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000/api"
).replace(/\/$/, "");

function resolveFigureUrl(value: string): string {
  if (/^https?:\/\//i.test(value)) {
    return value;
  }

  if (value.startsWith("/api/")) {
    return `${API_BASE_URL.replace(/\/api$/, "")}${value}`;
  }

  return `${API_BASE_URL}/${value.replace(/^\//, "")}`;
}

function FigureImage({
  figure,
  className,
  loading,
}: {
  figure: FigureReference;
  className: string;
  loading?: "eager" | "lazy";
}) {
  const originalUrl = resolveFigureUrl(figure.url);
  const previewUrl = figure.preview_url
    ? resolveFigureUrl(figure.preview_url)
    : null;
  const [source, setSource] = useState(previewUrl ?? originalUrl);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setSource(previewUrl ?? originalUrl);
    setFailed(false);
  }, [originalUrl, previewUrl]);

  if (failed) {
    return (
      <span className="text-xs font-medium text-slate-400">
        Image unavailable
      </span>
    );
  }

  return (
    <img
      src={source}
      alt={
        figure.title ??
        `${figure.document} p.${figure.page ?? "?"} figure`
      }
      loading={loading}
      className={className}
      onError={() => {
        if (previewUrl && source !== originalUrl) {
          setSource(originalUrl);
          return;
        }
        setFailed(true);
      }}
    />
  );
}

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponseWithFigures;
  comparison?: ChatCompareResponseWithFigures;
};

type SavedChat = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
};

const CHAT_HISTORY_STORAGE_KEY = "ai_brain_chat_history_v1";
const MAX_SAVED_CHATS = 30;

function createChatId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function makeChatTitle(messages: ChatMessage[]): string {
  const firstQuestion = messages.find(
    (message) => message.role === "user",
  )?.content;

  if (!firstQuestion) {
    return "이전 대화";
  }

  const compact = firstQuestion.replace(/\s+/g, " ").trim();
  return compact.length > 34
    ? `${compact.slice(0, 34)}...`
    : compact;
}

function formatChatDate(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleString("ko-KR", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function serializeChatMessages(messages: ChatMessage[]): string {
  return JSON.stringify(messages);
}

export default function Home() {
  const [documentFiles, setDocumentFiles] = useState<File[]>([]);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [savedChats, setSavedChats] = useState<SavedChat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const historyReadyRef = useRef(false);
  const [vision, setVision] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploadJob, setUploadJob] = useState<UploadJob | null>(null);
  const [expandedSource, setExpandedSource] = useState<string | null>(null);
  const [knowledgeRefreshKey, setKnowledgeRefreshKey] = useState(0);
  const [selectedFigure, setSelectedFigure] = useState<FigureReference | null>(null);
  const [answerMode, setAnswerMode] = useState<AnswerMode>("qwen3:8b");

  useEffect(() => {
    if (!selectedFigure) {
      return;
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSelectedFigure(null);
      }
    }

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [selectedFigure]);


  useEffect(() => {
    const nextChatId = createChatId();
    setActiveChatId(nextChatId);

    try {
      const rawHistory = window.localStorage.getItem(
        CHAT_HISTORY_STORAGE_KEY,
      );

      if (rawHistory) {
        const parsed = JSON.parse(rawHistory);

        if (Array.isArray(parsed)) {
          const validChats = parsed.filter(
            (chat): chat is SavedChat =>
              chat &&
              typeof chat.id === "string" &&
              typeof chat.title === "string" &&
              typeof chat.createdAt === "string" &&
              typeof chat.updatedAt === "string" &&
              Array.isArray(chat.messages),
          );

          setSavedChats(validChats.slice(0, MAX_SAVED_CHATS));
        }
      }
    } catch (error) {
      console.warn("대화 기록을 불러오지 못했습니다.", error);
    } finally {
      historyReadyRef.current = true;
    }
  }, []);

  useEffect(() => {
    if (
      !historyReadyRef.current ||
      !activeChatId ||
      messages.length === 0 ||
      !messages.some((message) => message.role === "user")
    ) {
      return;
    }

    setSavedChats((previous) => {
      const now = new Date().toISOString();
      const existing = previous.find(
        (chat) => chat.id === activeChatId,
      );
      const title = makeChatTitle(messages);
      const messageSnapshot = serializeChatMessages(messages);

      if (
        existing &&
        existing.title === title &&
        serializeChatMessages(existing.messages) === messageSnapshot
      ) {
        return previous;
      }

      const savedChat: SavedChat = {
        id: activeChatId,
        title,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
        messages,
      };

      const next = [
        savedChat,
        ...previous.filter((chat) => chat.id !== activeChatId),
      ]
        .sort(
          (left, right) =>
            new Date(right.updatedAt).getTime() -
            new Date(left.updatedAt).getTime(),
        )
        .slice(0, MAX_SAVED_CHATS);

      try {
        window.localStorage.setItem(
          CHAT_HISTORY_STORAGE_KEY,
          JSON.stringify(next),
        );
      } catch (error) {
        console.warn("대화 기록을 저장하지 못했습니다.", error);
      }

      return next;
    });
  }, [activeChatId, messages]);

  function startNewChat() {
    setActiveChatId(createChatId());
    setMessages([]);
    setQuestion("");
    setStatus(null);
    setExpandedSource(null);
    setVision(null);
    setSelectedFigure(null);
  }

  function openSavedChat(chat: SavedChat) {
    setActiveChatId(chat.id);
    setMessages(chat.messages);
    setQuestion("");
    setStatus(null);
    setExpandedSource(null);
    setVision(null);
    setSelectedFigure(null);
  }

  function deleteSavedChat(chatId: string) {
    const target = savedChats.find((chat) => chat.id === chatId);
    const confirmed = window.confirm(
      target
        ? `"${target.title}" 대화를 삭제할까요?`
        : "이 대화를 삭제할까요?",
    );

    if (!confirmed) {
      return;
    }

    setSavedChats((previous) => {
      const next = previous.filter((chat) => chat.id !== chatId);

      try {
        window.localStorage.setItem(
          CHAT_HISTORY_STORAGE_KEY,
          JSON.stringify(next),
        );
      } catch (error) {
        console.warn("대화 기록을 삭제하지 못했습니다.", error);
      }

      return next;
    });

    if (activeChatId === chatId) {
      startNewChat();
    }
  }

  function clearAllSavedChats() {
    if (savedChats.length === 0) {
      return;
    }

    const confirmed = window.confirm(
      `저장된 최근 대화 ${savedChats.length}개를 모두 삭제할까요?`,
    );

    if (!confirmed) {
      return;
    }

    try {
      window.localStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
    } catch (error) {
      console.warn("전체 대화 기록을 삭제하지 못했습니다.", error);
    }

    setSavedChats([]);
    startNewChat();
  }

  async function onUpload() {
    if (!documentFiles.length) return;
    setBusy(true);
    setUploadJob(null);

    let completed = 0;
    let skipped = 0;
    let totalChunks = 0;

    try {
      for (const file of documentFiles) {
        setStatus(`Uploading ${completed + 1}/${documentFiles.length}: ${file.name}`);
        const job = await createUploadJob();
        const poller = setInterval(async () => {
          try {
            setUploadJob(await getUploadJob(job.job_id));
          } catch {
            // Keep upload running even if one polling request fails.
          }
        }, 1000);

        try {
          const result = await uploadDocument(file, true, job.job_id);
          const finalJob = await getUploadJob(job.job_id);
          setUploadJob(finalJob);
          completed += 1;
          if (result.skipped) skipped += 1;
          totalChunks += result.document?.chunks ?? 0;
        } finally {
          clearInterval(poller);
        }
      }

      setStatus(`Uploaded ${completed}/${documentFiles.length} files · ${totalChunks} chunks${skipped ? ` · ${skipped} reused` : ""}.`);
      setDocumentFiles([]);
      setKnowledgeRefreshKey((previous) => previous + 1);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Upload failed");
      if (completed > 0) {
        setKnowledgeRefreshKey((previous) => previous + 1);
      }
    } finally {
      setBusy(false);
    }
  }

  async function onAsk(event: FormEvent) {
    event.preventDefault();

    const submittedQuestion = question.trim();
    if (!submittedQuestion || busy) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: submittedQuestion,
    };

    setMessages((previous) => [...previous, userMessage]);
    setQuestion("");
    setBusy(true);
    setStatus(
      answerMode === "compare"
        ? "한 번 검색한 동일 근거로 Qwen3와 Gemma4를 순차 비교 중..."
        : `검색 후 ${answerMode}에 질문 중...`,
    );

    try {
      let assistantMessage: ChatMessage;
      let sourceCount = 0;

      if (answerMode === "compare") {
        const result = (await askComparison(
          submittedQuestion,
          ["qwen3:8b", "gemma4:latest"],
        )) as ChatCompareResponseWithFigures;

        sourceCount = result.sources?.length ?? 0;
        assistantMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "",
          comparison: result,
        };
      } else {
        const result = (await askQuestion(
          submittedQuestion,
          answerMode,
        )) as ChatResponseWithFigures;

        sourceCount = result.sources?.length ?? 0;
        assistantMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: result.answer,
          response: result,
        };
      }

      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ]);
      setStatus(
        sourceCount
          ? "동일한 검색 근거를 사용해 답변을 생성했습니다."
          : "관련 문서 청크를 찾지 못했습니다.",
      );
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Question failed";

      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `오류: ${errorMessage}`,
        },
      ]);
      setStatus(errorMessage);
    } finally {
      setBusy(false);
    }
  }

  async function onAnalyzeImage() {
    if (!imageFile) return;
    setBusy(true);
    setStatus("Analyzing graph image with Qwen2.5-VL...");
    try {
      const result = await analyzeImage(imageFile);
      setVision(result.analysis);
      setStatus("Vision analysis saved to figure notes.");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Vision analysis failed");
    } finally {
      setBusy(false);
    }
  }


  return (
    <main className="flex h-screen overflow-hidden bg-[#eef1f6] text-slate-900 md:pl-[72px]">
      <AppIconRail />
      <aside className="hidden w-[456px] shrink-0 border-r border-slate-200 bg-[#f8fafc] md:flex md:flex-col">
        <div className="flex h-14 items-center gap-3 border-b border-slate-200 px-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-600 text-sm font-bold text-white">A</div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-bold">Petroleum RAG Agent</h1>
            <p className="truncate text-[11px] text-slate-500">Petroleum RAG Agent</p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-4">
          <button
            type="button"
            onClick={startNewChat}
            className="mb-4 flex w-full items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium shadow-sm hover:bg-slate-50"
          >
            <span>＋</span>
            New chat
          </button>

          <div className="space-y-5">
            <section>
              <div className="mb-2 flex items-center justify-between px-2">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  Recent Chats
                </p>
                {savedChats.length > 0 && (
                  <button
                    type="button"
                    onClick={clearAllSavedChats}
                    className="rounded-md px-2 py-1 text-[10px] font-medium text-slate-400 transition hover:bg-red-50 hover:text-red-600"
                    title="최근 대화 전체 삭제"
                  >
                    전체 삭제
                  </button>
                )}
              </div>
              <div className="space-y-1 text-sm">
                {savedChats.length === 0 ? (
                  <p className="rounded-lg px-3 py-2 text-xs text-slate-400">
                    아직 저장된 이전 대화가 없습니다.
                  </p>
                ) : (
                  savedChats.map((chat) => {
                    const isActive =
                      activeChatId === chat.id &&
                      messages.length > 0;

                    return (
                      <div
                        key={chat.id}
                        className={`group flex items-center rounded-lg transition ${
                          isActive
                            ? "bg-indigo-50"
                            : "hover:bg-white"
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => openSavedChat(chat)}
                          className={`min-w-0 flex-1 px-3 py-2 text-left ${
                            isActive
                              ? "font-medium text-indigo-700"
                              : "text-slate-600"
                          }`}
                          title={chat.title}
                        >
                          <span className="block truncate">
                            💬 {chat.title}
                          </span>
                          <span className="mt-0.5 block text-[10px] text-slate-400">
                            {formatChatDate(chat.updatedAt)}
                          </span>
                        </button>

                        <button
                          type="button"
                          onClick={() => deleteSavedChat(chat.id)}
                          className="mr-2 flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-xs text-slate-300 opacity-0 transition hover:bg-red-50 hover:text-red-600 group-hover:opacity-100 focus:opacity-100"
                          aria-label={`${chat.title} 대화 삭제`}
                          title="대화 삭제"
                        >
                          ✕
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            </section>

            <section>
              <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Documents</p>
              <div className="mb-2 rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm">
                <input
                  className="w-full bg-transparent text-xs outline-none placeholder:text-slate-400"
                  placeholder="Search documents..."
                  disabled
                />
              </div>
              <DocumentInfoPanel refreshKey={knowledgeRefreshKey} />
            </section>

            <section>
              <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Tools</p>
              <div className="space-y-2">
                <label className="block cursor-pointer rounded-xl border border-slate-200 bg-white p-3 text-sm shadow-sm hover:border-indigo-200">
                  <input
                    className="hidden"
                    type="file"
                    multiple
                    accept=".pdf,.txt,.png,.jpg,.jpeg,.ppt,.pptx"
                    onChange={(event) => setDocumentFiles(Array.from(event.target.files ?? []))}
                  />
                  <span className="font-semibold">Upload document</span>
                  <span className="mt-1 block truncate text-xs text-slate-500">{documentFiles.length ? `${documentFiles.length} files selected` : "PDF, TXT, PPT, image"}</span>
                </label>
                <button
                  disabled={!documentFiles.length || busy}
                  onClick={onUpload}
                  className="w-full rounded-xl bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Upload · Extract · Embed
                </button>

                <label className="block cursor-pointer rounded-xl border border-slate-200 bg-white p-3 text-sm shadow-sm hover:border-indigo-200">
                  <input
                    className="hidden"
                    type="file"
                    accept=".png,.jpg,.jpeg"
                    onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
                  />
                  <span className="font-semibold">Graph/Image analysis</span>
                  <span className="mt-1 block truncate text-xs text-slate-500">{imageFile?.name ?? "PNG or JPG graph"}</span>
                </label>
                <button
                  disabled={!imageFile || busy}
                  onClick={onAnalyzeImage}
                  className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Analyze Image
                </button>
              </div>
            </section>

            <section>
              <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">System</p>
              <SystemStatusPanel refreshKey={knowledgeRefreshKey} />
            </section>
          </div>
        </div>

        <div className="border-t border-slate-200 p-3">
          <div className="rounded-xl bg-white p-3 text-xs text-slate-500 shadow-sm">
            <p className="font-semibold text-slate-700">Local-first mode</p>
            <p className="mt-1">Ollama · ChromaDB · Source-grounded answers</p>
          </div>
        </div>

      </aside>

      <section className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4 md:px-6">
          <div className="flex items-center gap-3">
            <div>
              <h2 className="text-sm font-bold">Petroleum RAG Agent 7.0 ▾</h2>
              <p className="hidden text-xs text-slate-500 sm:block">Citation-grounded Q&A with local knowledge base</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-slate-500">
            <MobileModeTabs />
            <a
              href="/evaluation"
              className="rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-xs font-semibold text-sky-700 hover:bg-sky-100"
            >
              Benchmark
            </a>
            <a
              href="/review"
              className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 hover:bg-indigo-100"
            >
              Figure review
            </a>
            <select
              value={answerMode}
              onChange={(event) =>
                setAnswerMode(event.target.value as AnswerMode)
              }
              disabled={busy}
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-700 outline-none disabled:opacity-50"
              aria-label="답변 모델 선택"
            >
              <option value="qwen3:8b">Qwen3 8B</option>
              <option value="gemma4:latest">Gemma4</option>
              <option value="compare">Qwen3 vs Gemma4</option>
            </select>
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-100">☾</span>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto bg-[#f8fafc] px-4 py-6">
          <div className="mx-auto flex max-w-4xl flex-col gap-5">
            {messages.length === 0 && (
              <div className="flex gap-3">
                <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">
                  AI
                </div>
                <div className="max-w-[86%] rounded-3xl rounded-tl-md bg-white px-5 py-4 text-sm leading-7 text-slate-700 shadow-sm ring-1 ring-slate-200">
                  <p className="font-semibold text-slate-900">
                    Petroleum RAG Agent가 준비됐어.
                  </p>
                  <p className="mt-2">
                    왼쪽에서 PDF/TXT/PPT 자료를 업로드한 뒤, 아래 입력창에 질문하면 검색된 문서 근거를 바탕으로 답변해.
                  </p>
                </div>
              </div>
            )}

            {messages.map((message) => {
              if (message.role === "user") {
                return (
                  <div key={message.id} className="flex justify-end">
                    <div className="max-w-[78%] whitespace-pre-wrap rounded-3xl rounded-br-md bg-white px-4 py-3 text-sm text-slate-700 shadow-sm ring-1 ring-slate-200">
                      {message.content}
                    </div>
                  </div>
                );
              }

              const comparison = message.comparison;
              const response = message.response ?? comparison;
              const sourceCount = response?.sources?.length ?? 0;
              const figures = response?.figures ?? [];

              return (
                <div key={message.id} className="flex gap-3">
                  <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">
                    AI
                  </div>

                  <div
                    className={`${
                      comparison ? "w-full max-w-none" : "max-w-[86%]"
                    } rounded-3xl rounded-tl-md bg-white px-5 py-4 text-sm leading-7 text-slate-700 shadow-sm ring-1 ring-slate-200`}
                  >
                    {comparison ? (
                      <>
                        <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                          <span className="rounded-full bg-emerald-50 px-3 py-1 font-semibold text-emerald-700">
                            동일 검색 근거
                          </span>
                          <span>
                            retrieval{" "}
                            {comparison.retrieval_elapsed_seconds.toFixed(2)}s
                          </span>
                          <span>
                            모델은 RTX 3090 안정성을 위해 순차 실행
                          </span>
                        </div>

                        <div className="grid gap-4 lg:grid-cols-2">
                          {comparison.answers.map((item) => (
                            <section
                              key={`${message.id}:${item.model}`}
                              className="min-w-0 rounded-2xl border border-slate-200 bg-slate-50 p-4"
                            >
                              <div className="mb-3 flex items-center justify-between gap-3 border-b border-slate-200 pb-3">
                                <h3 className="font-bold text-slate-900">
                                  {item.model}
                                </h3>
                                <span className="shrink-0 rounded-full bg-white px-2 py-1 text-[11px] font-semibold text-slate-500 ring-1 ring-slate-200">
                                  {item.elapsed_seconds.toFixed(2)}s
                                </span>
                              </div>
                              <MarkdownMath content={item.answer} />
                            </section>
                          ))}
                        </div>
                      </>
                    ) : (
                      <>
                        {message.response?.model && (
                          <div className="mb-3 flex items-center gap-2 text-xs text-slate-500">
                            <span className="rounded-full bg-indigo-50 px-3 py-1 font-semibold text-indigo-700">
                              {message.response.model}
                            </span>
                            {message.response.elapsed_seconds != null && (
                              <span>
                                {message.response.elapsed_seconds.toFixed(2)}s
                              </span>
                            )}
                          </div>
                        )}
                        <MarkdownMath content={message.content} />
                      </>
                    )}


                    {figures.length > 0 && (
                      <div className="mt-5 border-t border-slate-100 pt-4">
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                          Related figures
                        </h3>

                        <div className="mt-3 grid gap-3 sm:grid-cols-2">
                          {figures.map((figure) => (
                            <button
                              key={`${message.id}:${figure.filename}`}
                              type="button"
                              onClick={() => setSelectedFigure(figure)}
                              className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 text-left transition hover:border-indigo-300 hover:shadow-md"
                            >
                              <div className="flex h-48 items-center justify-center bg-white p-2">
                                <FigureImage
                                  figure={figure}
                                  loading="lazy"
                                  className="max-h-full max-w-full object-contain"
                                />
                              </div>

                              <div className="border-t border-slate-200 px-3 py-2">
                                <p className="truncate text-xs font-semibold text-slate-800">
                                  {figure.title ?? "Retrieved figure"}
                                </p>
                                <p className="mt-1 truncate text-[11px] text-slate-500">
                                  {figure.document}
                                  {figure.page ? ` · p.${figure.page}` : ""}
                                  {figure.image_type
                                    ? ` · ${figure.image_type}`
                                    : ""}
                                </p>
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {sourceCount > 0 && (
                      <div className="mt-5 border-t border-slate-100 pt-4">
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                          Sources
                        </h3>

                        <ul className="mt-2 space-y-2">
                          {response?.sources.map((source) => {
                            const sourceKey = `${message.id}:${source.chunk_id}`;

                            return (
                              <li
                                key={sourceKey}
                                className="rounded-2xl border border-slate-200 bg-slate-50 p-3"
                              >
                                <button
                                  className="w-full text-left"
                                  onClick={() =>
                                    setExpandedSource(
                                      expandedSource === sourceKey
                                        ? null
                                        : sourceKey,
                                    )
                                  }
                                >
                                  <span className="font-semibold text-slate-800">
                                    {source.document}
                                  </span>

                                  {source.page ? (
                                    <span className="ml-1 text-slate-500">
                                      p.{source.page}
                                    </span>
                                  ) : null}

                                  <span className="ml-2 text-xs text-indigo-600">
                                    score{" "}
                                    {source.score?.toFixed(3) ?? "n/a"}
                                  </span>

                                  <span className="block text-xs text-slate-400">
                                    chunk {source.chunk_id}
                                  </span>
                                </button>

                                <p className="mt-2 text-xs leading-5 text-slate-500">
                                  {source.preview ?? source.excerpt}
                                </p>

                                {expandedSource === sourceKey && (
                                  <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded-xl bg-white p-3 text-xs leading-5 text-slate-700 ring-1 ring-slate-200">
                                    {source.excerpt}
                                  </pre>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {uploadJob && (
              <div className="mx-auto w-full max-w-3xl rounded-2xl border border-slate-200 bg-white p-4 text-sm shadow-sm">
                <div className="flex items-center justify-between text-slate-700">
                  <span>{uploadJob.status === "failed" ? "❌" : uploadJob.status === "completed" ? "✅" : "⏳"} {uploadJob.message}</span>
                  <span>{uploadJob.step}/{uploadJob.total_steps}</span>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full bg-indigo-600" style={{ width: `${Math.min(100, (uploadJob.step / uploadJob.total_steps) * 100)}%` }} />
                </div>
                {uploadJob.error && <p className="mt-2 text-red-500">{uploadJob.error}</p>}
              </div>
            )}

            {vision && (
              <div className="flex gap-3">
                <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-violet-600 text-xs font-bold text-white">V</div>
                <div className="max-w-[86%] whitespace-pre-wrap rounded-3xl rounded-tl-md bg-white px-5 py-4 text-sm leading-7 text-slate-700 shadow-sm ring-1 ring-slate-200">
                  {vision}
                </div>
              </div>
            )}

            {status && (
              <div className="mx-auto rounded-full border border-slate-200 bg-white px-4 py-2 text-xs text-slate-500 shadow-sm">
                {busy ? "⏳ " : "✅ "}{status}
              </div>
            )}

            <details className="rounded-2xl border border-slate-200 bg-white p-4 text-sm shadow-sm">
              <summary className="cursor-pointer font-semibold text-slate-700">Plot panel</summary>
              <div className="mt-4">
                <PlotPanel />
              </div>
            </details>
          </div>
        </div>

        <form onSubmit={onAsk} className="shrink-0 border-t border-slate-200 bg-white px-4 py-4">
          <div className="mx-auto flex max-w-4xl items-end gap-2 rounded-3xl border border-slate-200 bg-white px-4 py-3 shadow-lg shadow-slate-200/70">
            <span className="pb-2 text-slate-400">＋</span>
            <textarea
              className="max-h-40 min-h-[36px] flex-1 resize-none bg-transparent py-2 text-sm outline-none placeholder:text-slate-400"
              placeholder="Message Petroleum RAG Agent..."
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <button
              disabled={busy || !question.trim()}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-white shadow-sm transition disabled:cursor-not-allowed disabled:opacity-40"
              aria-label="Send message"
            >
              ↑
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-slate-400">Petroleum RAG Agent can make mistakes. Check retrieved sources and page numbers.</p>
        </form>
      </section>


      {selectedFigure && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 p-4"
          role="dialog"
          aria-modal="true"
          aria-label={selectedFigure.title ?? "Retrieved figure"}
          onClick={() => setSelectedFigure(null)}
        >
          <div
            className="relative flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              aria-label="Close figure"
              onClick={() => setSelectedFigure(null)}
              className="absolute right-3 top-3 z-10 flex h-9 w-9 items-center justify-center rounded-full bg-slate-900/75 text-xl text-white hover:bg-slate-900"
            >
              ×
            </button>

            <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto bg-slate-100 p-4">
              <FigureImage
                figure={selectedFigure}
                loading="eager"
                className="max-h-[78vh] max-w-full object-contain"
              />
            </div>

            <div className="flex items-center justify-between gap-4 border-t border-slate-200 px-5 py-4">
              <div className="min-w-0">
                <p className="truncate font-semibold text-slate-900">
                  {selectedFigure.title ?? "Retrieved figure"}
                </p>
                <p className="mt-1 truncate text-sm text-slate-500">
                  {selectedFigure.document}
                  {selectedFigure.page
                    ? ` · p.${selectedFigure.page}`
                    : ""}
                  {selectedFigure.image_type
                    ? ` · ${selectedFigure.image_type}`
                    : ""}
                </p>
              </div>
              <a
                href={resolveFigureUrl(selectedFigure.url)}
                target="_blank"
                rel="noreferrer"
                className="shrink-0 text-sm font-semibold text-indigo-600 hover:text-indigo-700"
              >
                View original
              </a>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
