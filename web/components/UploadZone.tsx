'use client'

import { useRef, useState, useCallback, DragEvent } from 'react'
import { UploadCloud } from 'lucide-react'

interface Props {
  onUpload: (file: File) => void
  disabled?: boolean
}

export default function UploadZone({ onUpload, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = useCallback(
    (file: File) => {
      if (file.type === 'application/pdf') onUpload(file)
    },
    [onUpload]
  )

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={[
        'flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-12 py-16 transition-all cursor-pointer select-none',
        dragging
          ? 'border-indigo-500 bg-indigo-50'
          : 'border-gray-300 bg-white hover:border-indigo-400 hover:bg-indigo-50/40',
        disabled ? 'pointer-events-none opacity-60' : '',
      ].join(' ')}
    >
      <div className="rounded-full bg-indigo-100 p-4">
        <UploadCloud className="h-8 w-8 text-indigo-600" />
      </div>
      <div className="text-center">
        <p className="text-lg font-semibold text-gray-800">
          Drop your PDF here
        </p>
        <p className="mt-1 text-sm text-gray-500">
          or click to browse — any mortgage document package
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) handleFile(file)
          e.target.value = ''
        }}
      />
    </div>
  )
}
