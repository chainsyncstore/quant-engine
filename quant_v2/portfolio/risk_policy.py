"""Portfolio exposure risk policy for v2 execution routing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyResult:
    """Risk-policy output with adjusted exposure targets."""

    exposures: dict[str, float]
    gross_exposure: float
    net_exposure: float
    constraints_applied: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioRiskPolicy:
    """Hard-cap portfolio policy (symbol, bucket, gross, and net)."""

    max_symbol_exposure_frac: float = 0.05
    max_gross_exposure_frac: float = 0.20
    max_net_exposure_frac: float = 0.10
    correlation_bucket_caps: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.max_symbol_exposure_frac <= 1.0:
            raise ValueError("max_symbol_exposure_frac must be in (0, 1]")
        if not 0.0 < self.max_gross_exposure_frac <= 1.0:
            raise ValueError("max_gross_exposure_frac must be in (0, 1]")
        if not 0.0 < self.max_net_exposure_frac <= 1.0:
            raise ValueError("max_net_exposure_frac must be in (0, 1]")
        if self.max_net_exposure_frac > self.max_gross_exposure_frac:
            raise ValueError("max_net_exposure_frac cannot exceed max_gross_exposure_frac")
        for bucket, cap in self.correlation_bucket_caps.items():
            if not 0.0 < cap <= 1.0:
                raise ValueError(f"Invalid cap for bucket {bucket}: {cap}")

    def apply(
        self,
        exposures: dict[str, float],
        *,
        bucket_map: dict[str, str] | None = None,
    ) -> PolicyResult:
        """Apply hard caps and return adjusted target exposures."""

        adjusted = {symbol: float(value) for symbol, value in exposures.items() if value != 0.0}
        constraints: list[str] = []

        # Per-symbol cap
        symbol_capped = False
        for symbol, value in list(adjusted.items()):
            if abs(value) > self.max_symbol_exposure_frac:
                adjusted[symbol] = self.max_symbol_exposure_frac if value > 0 else -self.max_symbol_exposure_frac
                symbol_capped = True
        if symbol_capped:
            constraints.append("symbol_cap")

        # Correlation bucket cap
        if bucket_map and self.correlation_bucket_caps:
            bucket_capped = False
            buckets = {bucket_map.get(symbol, "unmapped") for symbol in adjusted}
            for bucket in buckets:
                cap = self.correlation_bucket_caps.get(bucket)
                if cap is None:
                    continue
                bucket_symbols = [s for s in adjusted if bucket_map.get(s, "unmapped") == bucket]
                if not bucket_symbols:
                    continue
                bucket_gross = sum(abs(adjusted[s]) for s in bucket_symbols)
                if bucket_gross <= cap or bucket_gross == 0.0:
                    continue
                scale = cap / bucket_gross
                for symbol in bucket_symbols:
                    adjusted[symbol] *= scale
                bucket_capped = True
            if bucket_capped:
                constraints.append("bucket_cap")

        # Gross cap
        gross = sum(abs(v) for v in adjusted.values())
        if gross > self.max_gross_exposure_frac and gross > 0.0:
            scale = self.max_gross_exposure_frac / gross
            adjusted = {symbol: value * scale for symbol, value in adjusted.items()}
            constraints.append("gross_cap")

        # Net cap (reduce dominant side only)
        net = sum(adjusted.values())
        if abs(net) > self.max_net_exposure_frac:
            if net > 0:
                long_symbols = [s for s, v in adjusted.items() if v > 0]
                long_sum = sum(adjusted[s] for s in long_symbols)
                short_sum = sum(abs(v) for v in adjusted.values() if v < 0)
                target_long_sum = self.max_net_exposure_frac + short_sum
                if long_sum > 0 and target_long_sum >= 0:
                    scale = min(1.0, target_long_sum / long_sum)
                    for symbol in long_symbols:
                        adjusted[symbol] *= scale
            else:
                short_symbols = [s for s, v in adjusted.items() if v < 0]
                short_sum = sum(abs(adjusted[s]) for s in short_symbols)
                long_sum = sum(v for v in adjusted.values() if v > 0)
                target_short_sum = self.max_net_exposure_frac + long_sum
                if short_sum > 0 and target_short_sum >= 0:
                    scale = min(1.0, target_short_sum / short_sum)
                    for symbol in short_symbols:
                        adjusted[symbol] *= scale
            constraints.append("net_cap")

        gross_final = float(sum(abs(v) for v in adjusted.values()))
        net_final = float(sum(adjusted.values()))
        return PolicyResult(
            exposures=adjusted,
            gross_exposure=gross_final,
            net_exposure=net_final,
            constraints_applied=tuple(dict.fromkeys(constraints)),
        )
