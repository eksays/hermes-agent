import { beforeEach, describe, expect, it } from 'vitest'

import {
  $previewStatusBySession,
  clearPreviewArtifacts,
  dismissPreviewArtifact,
  previewCandidateFromPayload,
  recordPreviewArtifact
} from './preview-status'

beforeEach(() => $previewStatusBySession.set({}))

describe('previewCandidateFromPayload', () => {
  it('extracts previewable paths and localhost urls', () => {
    expect(previewCandidateFromPayload({ path: '/abs/index.html' })).toBe('/abs/index.html')
    expect(previewCandidateFromPayload({ url: 'http://localhost:5173/' })).toBe('http://localhost:5173/')
  })

  it('pulls the html path out of an inline diff', () => {
    expect(previewCandidateFromPayload({ inline_diff: '\u001b[0ma/demo.html -> b/demo.html\u001b[0m\n' })).toBe(
      'demo.html'
    )
  })

  it('ignores non-previewable targets', () => {
    expect(previewCandidateFromPayload({ path: '/abs/notes.txt' })).toBe('')
    expect(previewCandidateFromPayload({ url: 'https://example.com/' })).toBe('')
    expect(previewCandidateFromPayload({})).toBe('')
  })
})

describe('recordPreviewArtifact', () => {
  it('dedupes by target and keeps the newest last', () => {
    recordPreviewArtifact('s1', '/a/index.html', '/work')
    recordPreviewArtifact('s1', '/a/about.html', '/work')
    recordPreviewArtifact('s1', '/a/index.html', '/work')

    expect($previewStatusBySession.get().s1.map(i => i.id)).toEqual(['/a/about.html', '/a/index.html'])
  })

  it('caps the list and derives a label', () => {
    for (const n of [1, 2, 3, 4, 5]) {
      recordPreviewArtifact('s1', `/a/p${n}.html`, '/work')
    }

    const list = $previewStatusBySession.get().s1
    expect(list).toHaveLength(4)
    expect(list[0].id).toBe('/a/p2.html')
    expect(list[3].label).toBe('p5.html')
  })

  it('dismiss and clear remove rows', () => {
    recordPreviewArtifact('s1', '/a/index.html', '/work')
    recordPreviewArtifact('s1', '/a/about.html', '/work')
    dismissPreviewArtifact('s1', '/a/index.html')
    expect($previewStatusBySession.get().s1.map(i => i.id)).toEqual(['/a/about.html'])

    clearPreviewArtifacts('s1')
    expect($previewStatusBySession.get().s1).toBeUndefined()
  })
})
