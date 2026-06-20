'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { FileText, RotateCcw, ChevronDown, ChevronUp, Zap, Target } from 'lucide-react'
import UploadZone from '@/components/UploadZone'
import DocSidebar, { type DocSection } from '@/components/DocSidebar'

// PDF viewer must be client-only (pdfjs uses browser APIs)
const PdfViewer = dynamic(() => import('@/components/PdfViewer'), {
  ssr: false,
  loading: () => (
    <div className="flex flex-1 items-center justify-center bg-gray-100 text-gray-400">
      Loading viewer…
    </div>
  ),
})

type Status = 'idle' | 'uploading' | 'processing' | 'done' | 'error'
type Mode = 'vlm' | 'tesseract'

interface PipelineResult {
  totalPages: number
  documents: DocSection[]
}

export default function Home() {
  const [status, setStatus] = useState<Status>('idle')
  const [mode, setMode] = useState<Mode>('vlm')
  const [logs, setLogs] = useState<string[]>([])
  const [error, setError] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [result, setResult] = useState<PipelineResult | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [targetPage, setTargetPage] = useState<number | null>(null)
  const [logsExpanded, setLogsExpanded] = useState(true)

  const logsRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [logs])

  const handleUpload = useCallback(async (file: File) => {
    setStatus('uploading')
    setLogs([])
    setError('')
    setResult(null)
    setCurrentPage(1)
    setTargetPage(null)

    const formData = new FormData()
    formData.append('pdf', file)
    formData.append('mode', mode)

    let response: Response
    try {
      setStatus('processing')
      response = await fetch('/api/process', { method: 'POST', body: formData })
    } catch (e) {
      setStatus('error')
      setError('Network error — could not reach the server.')
      return
    }

    if (!response.body) {
      setStatus('error')
      setError('No response body from server.')
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const event = JSON.parse(line.slice(6))

          if (event.type === 'log' || event.type === 'status') {
            setLogs((prev) => [...prev, event.message])
          } else if (event.type === 'done') {
            setSessionId(event.sessionId)
            setResult(event.result)
            setStatus('done')
          } else if (event.type === 'error') {
            setError(event.message)
            setStatus('error')
          }
        } catch {
          // Malformed SSE line — skip
        }
      }
    }
  }, [])

  const reset = () => {
    setStatus('idle')
    setLogs([])
    setError('')
    setResult(null)
    setSessionId('')
  }

  // ── Idle / processing / error states ─────────────────────────────────────
  if (status !== 'done') {
    return (
      <div className="flex h-full flex-col">
        {/* Header */}
        <header className="flex items-center gap-3 border-b border-gray-200 bg-white px-6 py-4 shadow-sm">
          <div className="rounded-lg bg-indigo-600 p-1.5">
            <FileText className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-gray-900">InfrX Document Pipeline</h1>
            <p className="text-xs text-gray-500">Mortgage package classifier & segmenter</p>
          </div>
        </header>

        <div className="flex flex-1 flex-col items-center justify-center gap-6 overflow-hidden p-8">
          {status === 'idle' && (
            <div className="w-full max-w-lg flex flex-col gap-4">
              {/* Mode toggle */}
              <div className="flex rounded-xl border border-gray-200 bg-gray-50 p-1 gap-1">
                <button
                  onClick={() => setMode('vlm')}
                  className={[
                    'flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-all',
                    mode === 'vlm'
                      ? 'bg-white text-indigo-700 shadow-sm border border-indigo-100'
                      : 'text-gray-500 hover:text-gray-700',
                  ].join(' ')}
                >
                  <Target className="h-4 w-4" />
                  VLM + Tesseract — Accurate
                </button>
                <button
                  onClick={() => setMode('tesseract')}
                  className={[
                    'flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-all',
                    mode === 'tesseract'
                      ? 'bg-white text-amber-700 shadow-sm border border-amber-100'
                      : 'text-gray-500 hover:text-gray-700',
                  ].join(' ')}
                >
                  <Zap className="h-4 w-4" />
                  Tesseract — Fast
                </button>
              </div>
              <p className="text-center text-xs text-gray-400">
                {mode === 'vlm'
                  ? 'GPT-4o-mini vision + Tesseract triage — best accuracy, ~3 min for scanned pages'
                  : 'Tesseract OCR only — free & instant, no API cost, less accurate on scanned docs'}
              </p>
              <UploadZone onUpload={handleUpload} />
            </div>
          )}

          {(status === 'uploading' || status === 'processing') && (
            <div className="w-full max-w-2xl">
              <div className="mb-4 flex items-center gap-3">
                <svg className="h-5 w-5 animate-spin text-indigo-600" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
                <span className="font-medium text-gray-700">
                  {status === 'uploading'
                    ? 'Uploading PDF…'
                    : mode === 'vlm'
                    ? 'Running pipeline — VLM + Tesseract mode (accurate)…'
                    : 'Running pipeline — Tesseract-only mode (fast)…'}
                </span>
                <button
                  onClick={() => setLogsExpanded((v) => !v)}
                  className="ml-auto flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
                >
                  {logsExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  {logsExpanded ? 'Hide' : 'Show'} logs
                </button>
              </div>

              {logsExpanded && (
                <div
                  ref={logsRef}
                  className="h-96 overflow-y-auto rounded-xl bg-gray-950 p-4 font-mono text-xs text-green-400 shadow-inner"
                >
                  {logs.length === 0 ? (
                    <span className="text-gray-600">Waiting for pipeline output…</span>
                  ) : (
                    logs.map((line, i) => (
                      <div key={i} className="whitespace-pre-wrap leading-relaxed">
                        {line}
                      </div>
                    ))
                  )}
                  <div className="mt-1 animate-pulse text-gray-600">▌</div>
                </div>
              )}
            </div>
          )}

          {status === 'error' && (
            <div className="w-full max-w-lg">
              <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-center">
                <p className="mb-1 text-sm font-semibold text-red-700">Pipeline failed</p>
                <p className="mb-4 text-xs text-red-600">{error}</p>
                <button
                  onClick={reset}
                  className="inline-flex items-center gap-2 rounded-lg bg-red-100 px-4 py-2 text-sm font-medium text-red-700 hover:bg-red-200"
                >
                  <RotateCcw className="h-4 w-4" /> Try again
                </button>
              </div>

              {logs.length > 0 && (
                <div
                  className="mt-4 h-48 overflow-y-auto rounded-xl bg-gray-950 p-4 font-mono text-xs text-red-400"
                >
                  {logs.map((line, i) => (
                    <div key={i} className="whitespace-pre-wrap leading-relaxed">{line}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    )
  }

  // ── Done state: split view ────────────────────────────────────────────────
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-3 shadow-sm">
        <div className="rounded-lg bg-indigo-600 p-1.5">
          <FileText className="h-5 w-5 text-white" />
        </div>
        <div className="flex-1">
          <h1 className="text-sm font-bold text-gray-900">InfrX Document Pipeline</h1>
          <p className="text-xs text-gray-500">
            {result!.documents.length} documents detected · {result!.totalPages} pages · page {currentPage}
          </p>
        </div>
        <button
          onClick={reset}
          className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 shadow-sm hover:bg-gray-50"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          New PDF
        </button>
      </header>

      {/* Split view */}
      <div className="flex flex-1 overflow-hidden">
        <DocSidebar
          documents={result!.documents}
          currentPage={currentPage}
          onSelect={(startPage) => setTargetPage(startPage)}
        />
        <PdfViewer
          pdfUrl={`/api/pdf/${sessionId}`}
          totalPages={result!.totalPages}
          targetPage={targetPage}
          onPageChange={setCurrentPage}
        />
      </div>
    </div>
  )
}
