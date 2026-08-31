import { mulberry32, gaussian } from '../lib/rng'
import { fidelity, detectors } from '../data/run'

export type Band = 'approve' | 'step_up' | 'hold' | 'decline' | 'block'

export interface Authorization {
  id: number
  cardId: string
  ts: string
  amount: number
  category: string
  hour: number
  deviceAgeDays: number
  deviceNewToCard: boolean
  entryMode: string
  authsLastHour: number
  amountVsMedian: number
  withinUsualHours: boolean
  firstAtMerchant: boolean
  score: number
  band: Band
  contributions: Array<{ feature: string; value: string; weight: number }>
}

const ENTRY_MODES = ['chip', 'contactless', 'card_not_present', 'token']

/** Fitted von Mises mixture over hour of day, from the calibration artifact. */
function sampleHour(rand: () => number): number {
  const c = fidelity.circadian
  if (!c) return Math.floor(rand() * 24)
  const weights = c.weights as number[]
  const means = c.means as number[]
  const conc = c.concentrations as number[]
  const k = rand() < weights[0] ? 0 : 1
  // Wrapped normal approximation to von Mises: sd = 1/sqrt(kappa) in radians.
  const sdHours = (1 / Math.sqrt(conc[k])) * (24 / (2 * Math.PI))
  const h = means[k] + gaussian(rand) * sdHours
  return ((h % 24) + 24) % 24
}

function sampleCategory(rand: () => number): string {
  const mix = Object.entries(fidelity.category_mix ?? { retail: 1 })
  const r = rand()
  let acc = 0
  for (const [name, share] of mix) {
    acc += share
    if (r <= acc) return name
  }
  return mix[mix.length - 1][0]
}

/**
 * Sample one authorization from the fitted distributions and band it with the
 * real cost-curve thresholds.
 *
 * The distribution parameters and the band thresholds are measured. The
 * individual event is sampled, and the risk score below is a transparent
 * stand-in stated on screen, not the trained XGBoost ensemble. What this
 * demonstrates is the mitigation ladder and the shape of a scored event, which
 * is the part that has to fit an authorisation path.
 */
export function sampleAuthorization(seed: number, id: number): Authorization {
  const rand = mulberry32(seed + id * 2654435761)
  const a = fidelity.amount

  const amount = Math.exp(a.lognormal_mu + a.lognormal_sigma * gaussian(rand))
  const rounded = rand() < a.whole_number_share ? Math.round(amount) : Math.round(amount * 100) / 100
  const hour = sampleHour(rand)
  const category = sampleCategory(rand)

  const deviceAgeDays = Math.floor(Math.pow(rand(), 2.2) * 900)
  const deviceNewToCard = deviceAgeDays < 2 && rand() < 0.75
  const entryMode = ENTRY_MODES[Math.floor(rand() * ENTRY_MODES.length)]
  const authsLastHour = rand() < 0.9 ? Math.floor(rand() * 3) : 3 + Math.floor(rand() * 9)
  const amountVsMedian = rounded / a.median
  const withinUsualHours = hour > 7 && hour < 23
  const firstAtMerchant = rand() < 0.28

  // A transparent linear score over the features the real top-gain list is led
  // by: device age, velocity, entry mode, amount. Stated on screen so nobody
  // mistakes it for the trained model.
  const contributions = [
    {
      feature: 'device_age_days',
      value: deviceNewToCard ? `${deviceAgeDays} (new to card)` : String(deviceAgeDays),
      weight: deviceNewToCard ? 0.34 : deviceAgeDays < 30 ? 0.16 : -0.06,
    },
    {
      feature: 'auths_last_1h',
      value: String(authsLastHour),
      weight: authsLastHour >= 4 ? 0.26 : authsLastHour >= 2 ? 0.07 : -0.04,
    },
    {
      feature: 'entry_mode',
      value: entryMode,
      weight: entryMode === 'card_not_present' ? 0.19 : entryMode === 'token' ? -0.08 : -0.02,
    },
    {
      feature: 'amount_vs_median',
      value: amountVsMedian.toFixed(2),
      weight: amountVsMedian > 6 ? 0.21 : amountVsMedian > 2.5 ? 0.09 : -0.03,
    },
    {
      feature: 'within_usual_hours',
      value: String(withinUsualHours),
      weight: withinUsualHours ? -0.09 : 0.14,
    },
    {
      feature: 'is_first_txn_this_merchant',
      value: String(firstAtMerchant),
      weight: firstAtMerchant ? 0.08 : -0.02,
    },
  ]

  const raw = contributions.reduce((acc, c) => acc + c.weight, 0)
  const score = 1 / (1 + Math.exp(-(raw * 3.1)))

  const b = detectors.fitted_bands
  let band: Band = 'approve'
  if (b) {
    if (score >= b.block) band = 'block'
    else if (score >= b.decline) band = 'decline'
    else if (score >= b.hold) band = 'hold'
    else if (score >= b.step_up) band = 'step_up'
  }

  const hh = String(Math.floor(hour)).padStart(2, '0')
  const mm = String(Math.floor((hour % 1) * 60)).padStart(2, '0')
  const ss = String(Math.floor(rand() * 60)).padStart(2, '0')

  return {
    id,
    cardId: `c_${(Math.floor(rand() * 0xfffff) >>> 0).toString(16).padStart(5, '0')}`,
    ts: `${hh}:${mm}:${ss}`,
    amount: rounded,
    category,
    hour,
    deviceAgeDays,
    deviceNewToCard,
    entryMode,
    authsLastHour,
    amountVsMedian,
    withinUsualHours,
    firstAtMerchant,
    score,
    band,
    contributions,
  }
}

export const BAND_TONE: Record<Band, 'pass' | 'def' | 'value' | 'atk'> = {
  approve: 'pass',
  step_up: 'def',
  hold: 'value',
  decline: 'value',
  block: 'atk',
}

export const BAND_MITIGATION: Record<Band, string> = {
  approve: 'let it through',
  step_up: 'challenge the cardholder',
  hold: 'freeze the card 24h',
  decline: 'freeze 72h',
  block: 'unbind device, add to blocklist',
}
