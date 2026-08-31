"""End-to-end calibration.

Runs every fit against the judge dataset and writes one artifact. The swept
entries are recorded here alongside the fitted ones so the artifact is a
complete account of where each number came from, including the ones no data
settles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..logs import emit, get_logger
from ..paths import ARTIFACT_DIR
from .artifact import FittedParams
from .behavioral import fanout_stats
from .fit_amount import fit_amount
from .fit_arrival import fit_arrival, measure_targets, simulate_arrival
from .fit_circadian import fit_circadian, fit_hierarchical_circadian
from .fit_fanout import fit_fanout
from .fit_heterogeneity import fit_heterogeneity
from .fit_timing import fit_hawkes, sequences_from_frame
from .loaders import IeeeCisLoader, SparkovLoader
from .noise_floor import NoiseFloorBuilder

_log = get_logger(__name__)


def run_calibration(
    seed: int = 0,
    max_timing_entities: int = 1500,
    include_rejected_hawkes: bool = True,
    verbose: bool = True,
) -> FittedParams:
    """Fit every model and assemble the artifact."""

    def say(message: str) -> None:
        if verbose:
            _log.info("%s", message)

    say("loading judge dataset")
    frame = IeeeCisLoader().transactions()
    benign = frame.benign()

    say("computing noise floors")
    floors = NoiseFloorBuilder(
        benign, "entity", "TransactionDT", "TransactionAmt", seed=seed
    ).build()

    params = FittedParams(
        source="ieee-cis train_transaction, benign rows",
        split_fingerprint=floors.split_fingerprint,
        split_seed=seed,
    )
    params.noise_floors = dict(floors.floors)
    params.targets = dict(floors.targets)

    say("fitting amount")
    amount = fit_amount(benign["TransactionAmt"].to_numpy(float))
    params.add_fitted("amount", amount.as_dict())

    say("fitting amount heterogeneity")
    heterogeneity = fit_heterogeneity(benign, "entity", "TransactionAmt")
    params.add_fitted("amount_heterogeneity", heterogeneity.as_dict())

    say("fitting circadian")
    hours = (benign["TransactionDT"].to_numpy(float) / 3600.0) % 24
    circadian = fit_circadian(hours, n_components=2, seed=seed)
    params.add_fitted(
        "circadian",
        {
            "means": list(circadian.means),
            "concentrations": list(circadian.concentrations),
            "weights": list(circadian.weights),
            "resultant_length": circadian.resultant_length,
            "n_samples": circadian.n_samples,
        },
    )

    say("fitting per-holder circadian habits")
    benign_hours = benign.copy()
    benign_hours["_hour"] = (benign["TransactionDT"].to_numpy(float) / 3600.0) % 24
    hierarchical = fit_hierarchical_circadian(
        benign_hours, "entity", "_hour", min_events=10, seed=seed
    )
    params.add_fitted("circadian_hierarchical", hierarchical.as_dict())

    say("fitting arrival timing")
    sequences = sequences_from_frame(benign, "entity", "TransactionDT", min_events=10)
    arrival = fit_arrival(sequences, seed=seed)
    params.add_fitted("arrival", arrival.as_dict())

    observed = measure_targets(
        simulate_arrival(arrival, 2000, 30, np.random.default_rng(seed + 4242))
    )
    params.add_diagnostic(
        "arrival_holdout",
        {
            "note": "checked on a seed the search did not use",
            "autocorrelation_target": arrival.target_autocorrelation,
            "autocorrelation_observed": observed["autocorrelation"],
            "burstiness_target": arrival.target_burstiness,
            "burstiness_observed": observed["burstiness"],
        },
    )

    if include_rejected_hawkes:
        say("fitting hawkes, recorded as a rejected alternative")
        hawkes = fit_hawkes(sequences, max_entities=max_timing_entities, seed=seed)
        params.add_rejection(
            "hawkes",
            reason=(
                "failed its time-rescaling gate; the fitted decay of "
                f"{1 / hawkes.beta / 86400:.1f} days means the excitation term was "
                "absorbing the spread of per-entity rates rather than describing bursts"
            ),
            payload={
                "ks_pvalue": hawkes.ks_pvalue,
                "branching_ratio": hawkes.branching_ratio,
                "decay_seconds": 1.0 / hawkes.beta,
            },
        )

    say("fitting fan-out")
    joined = IeeeCisLoader().fingerprint_to_entity()
    benign_joined = joined[joined["isFraud"] == 0]
    degrees = fanout_stats(benign_joined, "fingerprint", "entity").degrees
    fanout = fit_fanout(degrees, seed=seed)
    params.add_fitted("fingerprint_fanout", fanout.as_dict())

    say("reading category taxonomy")
    sparkov = SparkovLoader()
    mix = sparkov.cluster_mix()
    params.add_fitted("category_mix", {str(k): float(v) for k, v in mix.items()})

    demographics = sparkov.demographics()
    ages = demographics["age_years"]
    params.add_fitted(
        "demographics",
        {
            "age_p10": float(ages.quantile(0.1)),
            "age_p50": float(ages.quantile(0.5)),
            "age_p90": float(ages.quantile(0.9)),
            "n_holders": len(demographics),
            "n_jobs": int(demographics["job"].nunique()),
        },
    )

    _record_swept(params)
    return params


def _record_swept(params: FittedParams) -> None:
    """Parameters no available data settles.

    Each carries the range it is swept over and the reason it cannot be fitted,
    so a later claim can be stated as holding across the range rather than at a
    point estimate that was never measured.
    """
    params.add_swept(
        "device_household_mean",
        value=2.1,
        low=1.5,
        high=4.0,
        reason=(
            "no source separates a physical device from a configuration fingerprint, "
            "so household-scale sharing is assumed rather than measured"
        ),
    )
    params.add_swept(
        "device_household_max",
        value=8,
        low=4,
        high=15,
        reason="upper bound on plausible household sharing of one device",
    )
    params.add_swept(
        "geo_home_radius_km",
        value=12.0,
        low=3.0,
        high=40.0,
        reason=(
            "the taxonomy source places merchants in a ring around each customer, with a "
            "tenth percentile distance of 35km and no local spend at all, so its geography "
            "cannot be fitted"
        ),
    )
    params.add_swept(
        "merchant_popularity_exponent",
        value=1.2,
        low=0.6,
        high=2.2,
        reason=(
            "the taxonomy source's merchant traffic is nearly flat, and the judge dataset "
            "carries no merchant entity at all"
        ),
    )
    params.add_swept(
        "merchant_loyalty",
        value=0.55,
        low=0.0,
        high=0.9,
        reason=(
            "nothing anchors it. The judge dataset carries no merchant entity, and the "
            "taxonomy source is itself a generator whose typical card visits 572 of its "
            "693 merchants with a top-1 share of 0.67%, so its per-card concentration of "
            "1.05x the marginal is geographic rather than loyalty. The low end reproduces "
            "the uniform draw this replaces, so the range contains the null"
        ),
    )
    params.add_swept(
        "merchant_preferred_set_mean",
        value=12.0,
        low=3.0,
        high=60.0,
        reason=(
            "how many merchants a cardholder uses regularly is unmeasured in both "
            "sources; the taxonomy source's cards have no regulars at all"
        ),
    )
    params.add_swept(
        "category_concentration",
        value=8.0,
        low=1.0,
        high=100.0,
        reason=(
            "the taxonomy source's per-card category concentration of 1.055x the marginal "
            "is significant against a shuffled null but tiny, and it is a generator whose "
            "category draw is close to a shared curve, so that figure is a lower bound "
            "rather than an estimate. High concentration reproduces the shared mix this "
            "replaces"
        ),
    )
    params.add_swept(
        "recovery_chain_probability",
        value=1e-5,
        low=1e-7,
        high=1e-4,
        reason=(
            "no published figure exists for how often a legitimate holder resets a "
            "password, calls support, and rebinds a device within an hour"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="fraudsim.calibration.pipeline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR / "fitted_params.json")
    parser.add_argument("--skip-hawkes", action="store_true",
                        help="skip refitting the rejected alternative")
    args = parser.parse_args(argv)

    params = run_calibration(seed=args.seed, include_rejected_hawkes=not args.skip_hawkes)
    emit()
    emit(params.render())
    emit(f"\nwrote {params.save(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
