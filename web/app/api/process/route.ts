import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import { writeFile, mkdir, readFile } from 'fs/promises'
import { join } from 'path'
import { randomUUID } from 'crypto'

// 10 minute timeout — pipeline can take 3-5 min for large PDFs
export const maxDuration = 600

const PROJECT_ROOT = join(process.cwd(), '..')

export async function POST(req: NextRequest) {
  const formData = await req.formData()
  const file = formData.get('pdf') as File | null
  const mode = (formData.get('mode') as string | null) ?? 'vlm'

  if (!file || file.type !== 'application/pdf') {
    return new Response(
      `data: ${JSON.stringify({ type: 'error', message: 'Please upload a valid PDF file.' })}\n\n`,
      { headers: { 'Content-Type': 'text/event-stream' } }
    )
  }

  const sessionId = randomUUID()
  const tmpDir = join(PROJECT_ROOT, 'tmp', sessionId)
  await mkdir(tmpDir, { recursive: true })

  const pdfPath = join(tmpDir, 'package.pdf')
  const outPath = join(tmpDir, 'pipeline_output.json')

  const bytes = await file.arrayBuffer()
  await writeFile(pdfPath, Buffer.from(bytes))

  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    start(controller) {
      const send = (data: object) => {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`))
        } catch {
          // controller already closed
        }
      }

      send({ type: 'status', message: 'Pipeline starting…', sessionId })

      const pythonScript = join(PROJECT_ROOT, 'src', 'pipeline', 'run_pipeline.py')
      const pipelineArgs = [pythonScript, '--pdf', pdfPath, '--out', outPath]
      if (mode === 'tesseract') pipelineArgs.push('--no-vlm')
      const proc = spawn(
        'python3',
        pipelineArgs,
        {
          cwd: PROJECT_ROOT,
          env: {
            ...process.env,
            OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? '',
          },
        }
      )

      let stderr = ''

      proc.stdout.on('data', (chunk: Buffer) => {
        const lines = chunk.toString().split('\n').filter(Boolean)
        for (const line of lines) {
          send({ type: 'log', message: line })
        }
      })

      proc.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString()
        const lines = chunk.toString().split('\n').filter(Boolean)
        for (const line of lines) {
          send({ type: 'log', message: line })
        }
      })

      proc.on('error', (err) => {
        send({ type: 'error', message: `Failed to start pipeline: ${err.message}` })
        controller.close()
      })

      proc.on('close', async (code) => {
        if (code !== 0) {
          send({ type: 'error', message: `Pipeline exited with code ${code}. ${stderr.slice(-300)}` })
          controller.close()
          return
        }

        try {
          const raw = JSON.parse(await readFile(outPath, 'utf-8'))

          // Send only the slim slice the UI needs — avoid shipping 12MB JSON
          const result = {
            totalPages: raw.total_pages as number,
            documents: (raw.documents as Array<{
              doc_instance_id: string
              doc_type: string
              start_page: number
              end_page: number
              page_count: number
              distinguishing_attr: string | null
            }>).map((d) => ({
              id: d.doc_instance_id,
              type: d.doc_type,
              startPage: d.start_page,
              endPage: d.end_page,
              pageCount: d.page_count,
              attr: d.distinguishing_attr,
            })),
            tables: (raw.tables as Array<{
              table_id: string
              doc_instance_id: string
              doctype: string
              page_span: { start_page: number; end_page: number }
              row_count_logical: number
              n_fragments: number
              columns: Array<{ col_idx: number }>
              cells: Array<{ row_idx: number; col_idx: number; text: string; is_header: boolean }>
            }>).map((t) => ({
              id: t.table_id,
              docId: t.doc_instance_id,
              docType: t.doctype,
              startPage: t.page_span?.start_page,
              endPage: t.page_span?.end_page,
              rows: t.row_count_logical,
              fragments: t.n_fragments,
              cols: t.columns?.length ?? 0,
              // Include first few rows of data for preview
              preview: t.cells
                ?.filter(c => c.is_header || c.row_idx >= 0)
                ?.slice(0, 20)
                ?.map(c => ({ r: c.row_idx, c: c.col_idx, t: c.text, h: c.is_header })) ?? [],
            })),
          }

          send({ type: 'done', sessionId, result })
        } catch (e) {
          send({ type: 'error', message: `Failed to read pipeline output: ${e}` })
        }

        controller.close()
      })
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  })
}
