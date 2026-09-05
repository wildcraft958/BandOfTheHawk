// Guards the single-file build. The catastrophic failure mode for a hosted
// artifact is silent: a blocked request renders nothing and reports nothing,
// so a broken page merely looks empty. Fail the build instead.
import { readFileSync, statSync } from 'node:fs'

const FILE = 'dist/index.html'
const ALLOW =
  /^https:\/\/(fonts\.googleapis\.com|fonts\.gstatic\.com|cdnjs\.cloudflare\.com|cdn\.jsdelivr\.net\/npm\/|code\.jquery\.com)/

const html = readFileSync(FILE, 'utf8')
const hard = []
const soft = []

for (const m of html.matchAll(/(?:src|href)\s*=\s*["'](https?:\/\/[^"']+)/gi)) {
  if (!ALLOW.test(m[1])) hard.push(`external ref: ${m[1]}`)
}
for (const m of html.matchAll(/url\(\s*['"]?(https?:\/\/[^)'"]+)/gi)) {
  if (!ALLOW.test(m[1])) hard.push(`css url(): ${m[1]}`)
}
for (const p of [/\bnew\s+Worker\b/, /\bnew\s+WebSocket\b/, /\bimportScripts\b/, /\bXMLHttpRequest\b/]) {
  if (p.test(html)) hard.push(`forbidden API: ${p}`)
}
if (/\bfetch\s*\(/.test(html)) soft.push('fetch( present - confirm it is unreachable vendor code')
if (/"_fixture"\s*:\s*true/.test(html)) soft.push('a synthetic fixture flag is bundled - banner must be visible')

const bytes = statSync(FILE).size
console.log(`check-csp: ${FILE} = ${(bytes / 1024).toFixed(0)} KB`)
for (const s of soft) console.warn(`  warn: ${s}`)
if (hard.length) {
  for (const h of hard) console.error(`  FAIL: ${h}`)
  process.exit(1)
}
if (bytes > 8 * 1024 * 1024) {
  console.error('  FAIL: exceeds 8 MB budget')
  process.exit(1)
}
console.log('check-csp: OK - self-contained, within budget')
