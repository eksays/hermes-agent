import { atom } from 'nanostores'

import { isPreviewableTarget } from '@/components/assistant-ui/tool-fallback-model'
import { previewName } from '@/lib/preview-targets'

/**
 * Session-scoped feed of previewable artifacts (HTML files, localhost dev URLs)
 * a tool produced. Surfaced as compact links in the composer status stack —
 * NOT auto-opened and NOT a bulky inline card. Click opens the rail preview or
 * the browser; both are manual.
 */
export interface PreviewArtifact {
  /** cwd captured at detection so a relative path still resolves on click. */
  cwd: string
  /** Dedupe key + display id (the raw target). */
  id: string
  label: string
  target: string
}

const MAX_PER_SESSION = 4

export const $previewStatusBySession = atom<Record<string, PreviewArtifact[]>>({})

const PREVIEW_FIELDS = ['url', 'target', 'path', 'file', 'filepath', 'preview'] as const

function stripAnsi(value: string): string {
  return value.replace(new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, 'g'), '')
}

function htmlPathFromInlineDiff(value: string): string {
  const cleaned = stripAnsi(value).replace(/^\s*┊\s*review diff\s*\n/i, '')

  for (const match of cleaned.matchAll(/(?:^|\s)(?:[ab]\/)?([^\s]+\.html?)(?=\s|$)/gi)) {
    const candidate = match[1]?.trim()

    if (candidate) {
      return candidate
    }
  }

  return ''
}

/** Pull a previewable target out of a `tool.*` event payload, or '' if none. */
export function previewCandidateFromPayload(payload: unknown): string {
  const record = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}

  for (const field of PREVIEW_FIELDS) {
    const value = record[field]

    if (typeof value === 'string' && isPreviewableTarget(value.trim())) {
      return value.trim()
    }
  }

  const inlineDiff = record.inline_diff

  return typeof inlineDiff === 'string' ? htmlPathFromInlineDiff(inlineDiff) : ''
}

const writePreviews = (sid: string, items: PreviewArtifact[]) => {
  const current = $previewStatusBySession.get()

  if (items.length === 0) {
    if (!current[sid]) {
      return
    }

    const next = { ...current }
    delete next[sid]
    $previewStatusBySession.set(next)

    return
  }

  $previewStatusBySession.set({ ...current, [sid]: items })
}

/** Record a detected artifact: dedupe by raw target, newest last, capped. */
export function recordPreviewArtifact(sid: string, target: string, cwd: string) {
  const raw = target.trim()

  if (!sid || !raw) {
    return
  }

  const list = $previewStatusBySession.get()[sid] ?? []
  const kept = list.filter(item => item.id !== raw)

  writePreviews(sid, [...kept, { cwd, id: raw, label: previewName(raw), target: raw }].slice(-MAX_PER_SESSION))
}

export function dismissPreviewArtifact(sid: string, id: string) {
  const list = $previewStatusBySession.get()[sid]

  if (list) {
    writePreviews(
      sid,
      list.filter(item => item.id !== id)
    )
  }
}

export function clearPreviewArtifacts(sid: string) {
  writePreviews(sid, [])
}
