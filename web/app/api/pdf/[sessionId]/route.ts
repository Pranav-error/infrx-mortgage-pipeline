import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'
import { existsSync } from 'fs'

const PROJECT_ROOT = join(process.cwd(), '..')

export async function GET(
  _req: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  const { sessionId } = params

  // Basic validation — sessionId should be a UUID
  if (!/^[0-9a-f-]{36}$/.test(sessionId)) {
    return NextResponse.json({ error: 'Invalid session' }, { status: 400 })
  }

  const pdfPath = join(PROJECT_ROOT, 'tmp', sessionId, 'package.pdf')

  if (!existsSync(pdfPath)) {
    return NextResponse.json({ error: 'PDF not found' }, { status: 404 })
  }

  const bytes = await readFile(pdfPath)

  return new Response(bytes, {
    headers: {
      'Content-Type': 'application/pdf',
      'Content-Disposition': 'inline',
      'Cache-Control': 'private, max-age=3600',
    },
  })
}
