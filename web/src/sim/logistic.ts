/**
 * Logistic regression with plain SGD.
 *
 * Not a stand-in for a "real" model: the defender's actual combiner is
 * LogisticRegression(max_iter=2000, class_weight="balanced") in
 * fraudsim/defender/combiner.py, and the binding expert is a LogisticRegression
 * too. This is the same family, scaled down, and it is really fitted by gradient
 * descent rather than animated.
 */
export class Logistic {
  readonly w: Float64Array
  b = 0
  fitted = false

  constructor(readonly dim: number) {
    this.w = new Float64Array(dim)
  }

  predict(x: Float64Array): number {
    let z = this.b
    for (let i = 0; i < this.dim; i++) z += this.w[i] * x[i]
    return 1 / (1 + Math.exp(-z))
  }

  /**
   * Class-weighted so the rare positive class is not ignored, which is the same
   * reason the real combiner passes class_weight="balanced" at a 0.5% base rate.
   */
  fit(rows: Float64Array[], labels: Uint8Array, epochs: number, lr: number): void {
    if (rows.length === 0) return
    const positives = labels.reduce((a, v) => a + v, 0)
    const negatives = rows.length - positives
    if (positives === 0 || negatives === 0) return

    const wPos = rows.length / (2 * positives)
    const wNeg = rows.length / (2 * negatives)

    for (let e = 0; e < epochs; e++) {
      for (let i = 0; i < rows.length; i++) {
        const x = rows[i]
        const y = labels[i]
        const p = this.predict(x)
        const g = (p - y) * (y === 1 ? wPos : wNeg)
        for (let d = 0; d < this.dim; d++) {
          this.w[d] -= lr * (g * x[d] + 1e-4 * this.w[d])
        }
        this.b -= lr * g
      }
    }
    this.fitted = true
  }

  /**
   * Balanced accuracy: the mean of recall on each class.
   *
   * Raw accuracy is meaningless here. The model is deliberately class-weighted
   * against a rare positive, so it over-predicts positives and raw accuracy
   * reads below 0.5 even when the model discriminates well.
   */
  balancedAccuracy(rows: Float64Array[], labels: Uint8Array): number {
    if (rows.length === 0) return 0
    let tp = 0
    let fn = 0
    let tn = 0
    let fp = 0
    for (let i = 0; i < rows.length; i++) {
      const hit = this.predict(rows[i]) >= 0.5
      if (labels[i] === 1) hit ? tp++ : fn++
      else hit ? fp++ : tn++
    }
    const recallPos = tp + fn > 0 ? tp / (tp + fn) : 0
    const recallNeg = tn + fp > 0 ? tn / (tn + fp) : 0
    return (recallPos + recallNeg) / 2
  }
}
