'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

// Use local worker copy (public/pdf.worker.min.mjs) — CDN has version/extension mismatch issues
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs'

interface Props {
  pdfUrl: string
  totalPages: number
  targetPage: number | null    // 0-indexed pipeline page to jump to
  onPageChange: (page: number) => void  // 1-indexed, reports current visible page
}

// Each rendered page is ~1.2MB canvas. We keep rendered pages in memory but
// only paint them when they've entered the viewport (lazy via IntersectionObserver).
function LazyPage({
  pageNumber,
  width,
  containerRef,
  pageRef,
  onVisible,
}: {
  pageNumber: number
  width: number
  containerRef: React.RefObject<HTMLDivElement>
  pageRef: (el: HTMLDivElement | null) => void
  onVisible: (n: number) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [shouldRender, setShouldRender] = useState(false)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShouldRender(true)
          onVisible(pageNumber)
        }
      },
      { root: containerRef.current, threshold: 0.05 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [pageNumber, containerRef, onVisible])

  return (
    <div
      ref={(el) => {
        ;(wrapRef as React.MutableRefObject<HTMLDivElement | null>).current = el
        pageRef(el)
      }}
      className="pdf-page-wrapper relative mb-3 overflow-hidden rounded shadow-md"
      style={{ minHeight: shouldRender ? undefined : Math.round((width * 11) / 8.5) }}
    >
      {/* Page number badge */}
      <div className="absolute left-2 top-2 z-10 rounded bg-black/40 px-1.5 py-0.5 text-xs font-mono text-white">
        {pageNumber}
      </div>

      {shouldRender ? (
        <Page
          pageNumber={pageNumber}
          width={width}
          renderTextLayer
          renderAnnotationLayer
          loading={
            <div
              className="animate-pulse bg-gray-200"
              style={{ height: Math.round((width * 11) / 8.5) }}
            />
          }
        />
      ) : (
        <div
          className="animate-pulse bg-gray-200"
          style={{ height: Math.round((width * 11) / 8.5) }}
        />
      )}
    </div>
  )
}

export default function PdfViewer({ pdfUrl, totalPages, targetPage, onPageChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map())
  const [numPages, setNumPages] = useState<number>(0)
  const [containerWidth, setContainerWidth] = useState(700)
  const prevTargetRef = useRef<number | null>(null)

  // Measure container width
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const observer = new ResizeObserver(([entry]) => {
      const w = Math.min(entry.contentRect.width - 32, 900)
      setContainerWidth(w)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Scroll to targetPage when it changes (0-indexed → 1-indexed for PDF)
  useEffect(() => {
    if (targetPage === null || targetPage === prevTargetRef.current) return
    prevTargetRef.current = targetPage

    const pdfPage = targetPage + 1  // convert 0-indexed to 1-indexed
    // Small delay to allow the page to render if not yet visible
    setTimeout(() => {
      const el = pageRefs.current.get(pdfPage)
      el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 100)
  }, [targetPage])

  const onVisible = useCallback(
    (pageNumber: number) => {
      onPageChange(pageNumber)
    },
    [onPageChange]
  )

  const setPageRef = useCallback(
    (pageNumber: number) => (el: HTMLDivElement | null) => {
      if (el) pageRefs.current.set(pageNumber, el)
      else pageRefs.current.delete(pageNumber)
    },
    []
  )

  return (
    <div
      ref={containerRef}
      className="flex h-full flex-1 flex-col overflow-y-auto bg-gray-100 px-4 py-4"
    >
      <Document
        file={pdfUrl}
        onLoadSuccess={({ numPages }) => setNumPages(numPages)}
        loading={
          <div className="flex h-full items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-gray-500">
              <svg className="h-8 w-8 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
              </svg>
              <span className="text-sm">Loading PDF…</span>
            </div>
          </div>
        }
        error={
          <div className="flex h-full items-center justify-center text-red-500">
            Failed to load PDF.
          </div>
        }
      >
        {Array.from({ length: numPages || totalPages }, (_, i) => i + 1).map((pageNum) => (
          <LazyPage
            key={pageNum}
            pageNumber={pageNum}
            width={containerWidth}
            containerRef={containerRef}
            pageRef={setPageRef(pageNum)}
            onVisible={onVisible}
          />
        ))}
      </Document>
    </div>
  )
}
