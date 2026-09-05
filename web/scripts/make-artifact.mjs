// Repackage the single-file build for a host that supplies its own document
// skeleton. Emits dist/artifact.html: the head's own tags plus the body's
// contents, with no doctype, html, head or body wrapper, so nothing nests.
import { readFileSync, writeFileSync } from 'node:fs'

const src = readFileSync('dist/index.html', 'utf8')

const head = src.match(/<head>([\s\S]*?)<\/head>/)?.[1] ?? ''
const body = src.match(/<body>([\s\S]*?)<\/body>/)?.[1] ?? ''

// charset and viewport come from the host skeleton.
const headKept = head
  .replace(/<meta\s+charset[^>]*>/gi, '')
  .replace(/<meta\s+name="viewport"[^>]*>/gi, '')
  .trim()

const out = `${headKept}\n${body.trim()}\n`

for (const tag of ['<!doctype', '<html', '<head', '<body']) {
  if (out.toLowerCase().includes(tag)) {
    console.error(`FAIL: ${tag} survived into the artifact payload`)
    process.exit(1)
  }
}

writeFileSync('dist/artifact.html', out)
const kb = (Buffer.byteLength(out) / 1024).toFixed(0)
console.log(`make-artifact: dist/artifact.html = ${kb} KB`)
console.log(`  title: ${out.match(/<title>(.*?)<\/title>/)?.[1] ?? 'MISSING'}`)
console.log(`  inline scripts: ${(out.match(/<script/g) || []).length}`)
console.log(`  inline styles:  ${(out.match(/<style/g) || []).length}`)
