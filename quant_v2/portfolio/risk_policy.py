"""Portfolio exposure risk policy for v2 execution routing."""

from __future__ import annotations

from dataclasses import dataclass, field


RISK_POLICY_VERSION = "wp03-risk-v1"


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

    max_symbol_exposure_frac: float = 0.15
    max_gross_exposure_frac: float = 1.0
    max_net_exposure_frac: float = 0.50
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

    def scaled(self, scale: float) -> "PortfolioRiskPolicy":
        """Return a downward-scaled operating policy derived from immutable limits."""

        scale = float(scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale must be within (0, 1]")
        return PortfolioRiskPolicy(
            max_symbol_exposure_frac=self.max_symbol_exposure_frac * scale,
            max_gross_exposure_frac=self.max_gross_exposure_frac * scale,
            max_net_exposure_frac=self.max_net_exposure_frac * scale,
            correlation_bucket_caps={
                bucket: float(limit) * scale
                for bucket, limit in self.correlation_bucket_caps.items()
            },
        )

    def clamp_down(
        self,
        *,
        max_symbol_exposure_frac: float | None = None,
        max_gross_exposure_frac: float | None = None,
        max_net_exposure_frac: float | None = None,
        correlation_bucket_caps: dict[str, float] | None = None,
    ) -> "PortfolioRiskPolicy":
        """Return an operating policy whose limits can only move downward."""

        symbol_cap = self.max_symbol_exposure_frac
        if max_symbol_exposure_frac is not None:
            symbol_cap = min(symbol_cap, float(max_symbol_exposure_frac))

        gross_cap = self.max_gross_exposure_frac
        if max_gross_exposure_frac is not None:
            gross_cap = min(gross_cap, float(max_gross_exposure_frac))

        net_cap = self.max_net_exposure_frac
        if max_net_exposure_frac is not None:
            net_cap = min(net_cap, float(max_net_exposure_frac))

        bucket_caps = {
            bucket: float(limit)
            for bucket, limit in self.correlation_bucket_caps.items()
        }
        if correlation_bucket_caps is not None:
            bucket_caps = {
                bucket: min(bucket_caps.get(bucket, float(limit)), float(limit))
                for bucket, limit in correlation_bucket_caps.items()
            }

        return PortfolioRiskPolicy(
            max_symbol_exposure_frac=symbol_cap,
            max_gross_exposure_frac=gross_cap,
            max_net_exposure_frac=min(net_cap, gross_cap),
            correlation_bucket_caps=bucket_caps,
        )


@dataclass(frozen=True)
class HardRiskLimits(PortfolioRiskPolicy):
    """Immutable ceiling limits used for breach detection and operating clamps."""

    policy_version: str = RISK_POLICY_VERSION


@dataclass(frozen=True)
class OperatingRiskLimits(PortfolioRiskPolicy):
    """Downward-only operating limits derived from immutable hard ceilings."""

    hard_limits: HardRiskLimits = field(default_factory=HardRiskLimits)
    policy_version: str = RISK_POLICY_VERSION
    limit_source: str = "static"
    target_headroom_ratio: float = 0.85
    fee_reserve_frac: float = 0.0
    slippage_reserve_frac: float = 0.0
    rounding_reserve_frac: float = 0.0
    min_quantity_reserve_frac: float = 0.0
    adverse_mark_buffer_frac: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "fee_reserve_frac",
            "slippage_reserve_frac",
            "rounding_reserve_frac",
            "min_quantity_reserve_frac",
            "adverse_mark_buffer_frac",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be within [0, 1)")
        if not 0.0 < float(self.target_headroom_ratio) <= 1.0:
            raise ValueError("target_headroom_ratio must be within (0, 1]")
        if self.policy_version != self.hard_limits.policy_version:
            raise ValueError("Operating policy version must match hard-limit version")
        if self.max_symbol_exposure_frac > self.hard_limits.max_symbol_exposure_frac:
            raise ValueError("Operating symbol cap cannot exceed hard symbol cap")
        if self.max_gross_exposure_frac > self.hard_limits.max_gross_exposure_frac:
            raise ValueError("Operating gross cap cannot exceed hard gross cap")
        if self.max_net_exposure_frac > self.hard_limits.max_net_exposure_frac:
            raise ValueError("Operating net cap cannot exceed hard net cap")
        for bucket, limit in self.correlation_bucket_caps.items():
            hard_limit = self.hard_limits.correlation_bucket_caps.get(bucket, float(limit))
            if float(limit) > float(hard_limit):
                raise ValueError(f"Operating bucket cap cannot exceed hard bucket cap for {bucket}")

    @property
    def reserve_capacity_frac(self) -> float:
        reserve = (
            float(self.fee_reserve_frac)
            + float(self.slippage_reserve_frac)
            + float(self.rounding_reserve_frac)
            + float(self.min_quantity_reserve_frac)
        )
        return min(max(reserve, 0.0), 0.95)

    @property
    def effective_headroom_ratio(self) -> float:
        return float(self.target_headroom_ratio) * (1.0 - self.reserve_capacity_frac)

    @property
    def headroom_frac(self) -> float:
        return max(0.0, 1.0 - self.effective_headroom_ratio)

    @classmethod
    def from_hard_limits(
        cls,
        hard_limits: HardRiskLimits,
        *,
        max_symbol_exposure_frac: float | None = None,
        max_gross_exposure_frac: float | None = None,
        max_net_exposure_frac: float | None = None,
        correlation_bucket_caps: dict[str, float] | None = None,
        limit_source: str = "static",
        target_headroom_ratio: float = 0.85,
        fee_reserve_frac: float = 0.0,
        slippage_reserve_frac: float = 0.0,
        rounding_reserve_frac: float = 0.0,
        min_quantity_reserve_frac: float = 0.0,
        adverse_mark_buffer_frac: float = 0.0,
    ) -> "OperatingRiskLimits":
        clamped = hard_limits.clamp_down(
            max_symbol_exposure_frac=max_symbol_exposure_frac,
            max_gross_exposure_frac=max_gross_exposure_frac,
            max_net_exposure_frac=max_net_exposure_frac,
            correlation_bucket_caps=correlation_bucket_caps,
        )
        return cls(
            max_symbol_exposure_frac=clamped.max_symbol_exposure_frac,
            max_gross_exposure_frac=clamped.max_gross_exposure_frac,
            max_net_exposure_frac=clamped.max_net_exposure_frac,
            correlation_bucket_caps=clamped.correlation_bucket_caps,
            hard_limits=hard_limits,
            policy_version=hard_limits.policy_version,
            limit_source=limit_source,
            target_headroom_ratio=target_headroom_ratio,
            fee_reserve_frac=fee_reserve_frac,
            slippage_reserve_frac=slippage_reserve_frac,
            rounding_reserve_frac=rounding_reserve_frac,
            min_quantity_reserve_frac=min_quantity_reserve_frac,
            adverse_mark_buffer_frac=adverse_mark_buffer_frac,
        )

    def scaled(self, scale: float, *, limit_source: str | None = None) -> "OperatingRiskLimits":
        scale = float(scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale must be within (0, 1]")
        return OperatingRiskLimits.from_hard_limits(
            self.hard_limits,
            max_symbol_exposure_frac=self.max_symbol_exposure_frac * scale,
            max_gross_exposure_frac=self.max_gross_exposure_frac * scale,
            max_net_exposure_frac=self.max_net_exposure_frac * scale,
            correlation_bucket_caps={
                bucket: float(limit) * scale
                for bucket, limit in self.correlation_bucket_caps.items()
            },
            limit_source=limit_source or self.limit_source,
            target_headroom_ratio=self.target_headroom_ratio,
            fee_reserve_frac=self.fee_reserve_frac,
            slippage_reserve_frac=self.slippage_reserve_frac,
            rounding_reserve_frac=self.rounding_reserve_frac,
            min_quantity_reserve_frac=self.min_quantity_reserve_frac,
            adverse_mark_buffer_frac=self.adverse_mark_buffer_frac,
        )


def build_dynamic_operating_limits(
    *,
    hard_limits: HardRiskLimits,
    sigma_60: float,
    limit_source: str = "dynamic_volatility",
    target_headroom_ratio: float = 0.85,
    fee_reserve_frac: float = 0.0,
    slippage_reserve_frac: float = 0.0,
    rounding_reserve_frac: float = 0.0,
    min_quantity_reserve_frac: float = 0.0,
    adverse_mark_buffer_frac: float = 0.0,
) -> OperatingRiskLimits | None:
    """Derive dynamic operating limits that can only tighten hard ceilings."""

    sigma_60 = float(sigma_60)
    if sigma_60 <= 1e-9:
        return None

    gross_cap = min(1.20 / sigma_60, 0.85)
    gross_cap = max(gross_cap, 0.05)
    net_cap = min(0.45 * gross_cap, gross_cap)
    symbol_cap = min(0.30 * gross_cap, 1.0)
    return OperatingRiskLimits.from_hard_limits(
        hard_limits,
        max_symbol_exposure_frac=symbol_cap,
        max_gross_exposure_frac=gross_cap,
        max_net_exposure_frac=net_cap,
        correlation_bucket_caps=hard_limits.correlation_bucket_caps,
        limit_source=limit_source,
        target_headroom_ratio=target_headroom_ratio,
        fee_reserve_frac=fee_reserve_frac,
        slippage_reserve_frac=slippage_reserve_frac,
        rounding_reserve_frac=rounding_reserve_frac,
        min_quantity_reserve_frac=min_quantity_reserve_frac,
        adverse_mark_buffer_frac=adverse_mark_buffer_frac,
    )
