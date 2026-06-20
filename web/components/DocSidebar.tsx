'use client'

import { useMemo } from 'react'
import clsx from 'clsx'

export interface DocSection {
  id: string
  type: string
  startPage: number
  endPage: number
  pageCount: number
  attr: string | null
}

interface Props {
  documents: DocSection[]
  currentPage: number        // 1-indexed (PDF page currently visible)
  onSelect: (startPage: number) => void
}

// Human-readable labels for doc types
const TYPE_LABELS: Record<string, string> = {
  form_1008: 'Form 1008',
  closing_disclosure: 'Closing Disclosure',
  purchase_contract: 'Purchase Contract',
  insurance_declaration: 'Insurance Declaration',
  filler: 'Filler / Blank',
  email_correspondence: 'Email',
  urla_1003: 'URLA 1003',
  loan_summary: 'Loan Summary',
  loan_estimate: 'Loan Estimate',
  paystub: 'Pay Stub',
  gift_letter: 'Gift Letter',
  bank_stmt_checking: 'Bank Statement',
  w2: 'W-2',
  voe: 'Verification of Employment',
  credit_report: 'Credit Report',
  deposit_receipt: 'Deposit Receipt',
  lpa_feedback: 'LPA Feedback',
  du_findings: 'DU Findings',
  letter_of_explanation: 'Letter of Explanation',
  narrative_chapter: 'Narrative Chapter',
  purchase_addendum: 'Purchase Addendum',
  check_image: 'Check Image',
  brokerage_stmt: 'Brokerage Statement',
}

// Color palette keyed by doc type
const TYPE_COLORS: Record<string, { dot: string; badge: string; active: string }> = {
  bank_stmt_checking:    { dot: 'bg-blue-500',    badge: 'bg-blue-100 text-blue-800',    active: 'bg-blue-50 border-blue-400' },
  urla_1003:             { dot: 'bg-violet-500',  badge: 'bg-violet-100 text-violet-800', active: 'bg-violet-50 border-violet-400' },
  paystub:               { dot: 'bg-emerald-500', badge: 'bg-emerald-100 text-emerald-800', active: 'bg-emerald-50 border-emerald-400' },
  closing_disclosure:    { dot: 'bg-amber-500',   badge: 'bg-amber-100 text-amber-800',   active: 'bg-amber-50 border-amber-400' },
  purchase_contract:     { dot: 'bg-rose-500',    badge: 'bg-rose-100 text-rose-800',     active: 'bg-rose-50 border-rose-400' },
  w2:                    { dot: 'bg-pink-500',     badge: 'bg-pink-100 text-pink-800',     active: 'bg-pink-50 border-pink-400' },
  credit_report:         { dot: 'bg-teal-500',    badge: 'bg-teal-100 text-teal-800',     active: 'bg-teal-50 border-teal-400' },
  loan_estimate:         { dot: 'bg-orange-500',  badge: 'bg-orange-100 text-orange-800', active: 'bg-orange-50 border-orange-400' },
  gift_letter:           { dot: 'bg-fuchsia-500', badge: 'bg-fuchsia-100 text-fuchsia-800', active: 'bg-fuchsia-50 border-fuchsia-400' },
  voe:                   { dot: 'bg-cyan-500',    badge: 'bg-cyan-100 text-cyan-800',     active: 'bg-cyan-50 border-cyan-400' },
  filler:                { dot: 'bg-gray-400',    badge: 'bg-gray-100 text-gray-600',     active: 'bg-gray-50 border-gray-300' },
}

function colorFor(type: string) {
  return TYPE_COLORS[type] ?? { dot: 'bg-indigo-500', badge: 'bg-indigo-100 text-indigo-800', active: 'bg-indigo-50 border-indigo-400' }
}

function labelFor(type: string) {
  return TYPE_LABELS[type] ?? type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function pageRange(doc: DocSection) {
  if (doc.startPage === doc.endPage) return `p${doc.startPage + 1}`
  return `p${doc.startPage + 1}–p${doc.endPage + 1}`
}

export default function DocSidebar({ documents, currentPage, onSelect }: Props) {
  // currentPage is 1-indexed; startPage/endPage are 0-indexed
  const activeIdx = useMemo(() => {
    const zeroPage = currentPage - 1
    return documents.findIndex(
      (d) => zeroPage >= d.startPage && zeroPage <= d.endPage
    )
  }, [documents, currentPage])

  // Summary stats
  const typeCounts = useMemo(() => {
    const map: Record<string, number> = {}
    for (const d of documents) map[d.type] = (map[d.type] ?? 0) + 1
    return map
  }, [documents])

  return (
    <aside className="flex h-full w-72 flex-shrink-0 flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-700 px-4 py-4">
        <p className="text-xs font-semibold uppercase tracking-widest text-gray-400">Sections</p>
        <p className="mt-1 text-sm text-gray-300">
          {documents.length} documents · {Object.keys(typeCounts).length} types
        </p>
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {documents.map((doc, idx) => {
          const col = colorFor(doc.type)
          const isActive = idx === activeIdx
          return (
            <button
              key={doc.id}
              onClick={() => onSelect(doc.startPage)}
              className={clsx(
                'group mb-1 flex w-full items-start gap-3 rounded-lg border px-3 py-2.5 text-left transition-all',
                isActive
                  ? 'border-gray-500 bg-gray-700'
                  : 'border-transparent hover:border-gray-600 hover:bg-gray-800'
              )}
            >
              {/* Color dot */}
              <span className={clsx('mt-1 h-2.5 w-2.5 flex-shrink-0 rounded-full', col.dot)} />

              <div className="min-w-0 flex-1">
                <p className={clsx(
                  'truncate text-sm font-medium',
                  isActive ? 'text-white' : 'text-gray-200 group-hover:text-white'
                )}>
                  {labelFor(doc.type)}
                  {doc.attr && (
                    <span className="ml-1.5 text-xs font-normal text-gray-400">
                      [{doc.attr}]
                    </span>
                  )}
                </p>
                <p className="mt-0.5 text-xs text-gray-500">
                  {pageRange(doc)} · {doc.pageCount} {doc.pageCount === 1 ? 'page' : 'pages'}
                </p>
              </div>

              {/* Ordinal badge */}
              <span className="flex-shrink-0 rounded px-1.5 py-0.5 text-xs font-mono text-gray-500">
                #{idx + 1}
              </span>
            </button>
          )
        })}
      </div>

      {/* Type legend */}
      <div className="border-t border-gray-700 px-4 py-3">
        <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-gray-500">Types</p>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(typeCounts).map(([type, count]) => {
            const col = colorFor(type)
            return (
              <span
                key={type}
                className={clsx('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium', col.badge)}
              >
                {labelFor(type)} {count > 1 && <span className="opacity-70">×{count}</span>}
              </span>
            )
          })}
        </div>
      </div>
    </aside>
  )
}
