"""Versioned execution cost policy for research, shadow, and paper paths."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

_DEFAULT_POLICY_VERSION = "wp07-execution-cost-v1"
_SCENARIO_MULTIPLIERS = {
    "base": 1.0,
    "adverse": 1.5,
    "severe": 2.0,
}

_FALLBACK_ADV_USD: dict[str, float] = {
    "BTCUSDT": 1_500_000_000.0,
    "ETHUSDT": 600_000_000.0,
    "BNBUSDT": 80_000_000.0,
    "XRPUSDT": 150_000_000.0,
    "SOLUSDT": 200_000_000.0,
    "ADAUSDT": 60_000_000.0,
    "DOGEUSDT": 80_000_000.0,
    "AVAXUSDT": 80_000_000.0,
    "LINKUSDT": 50_000_000.0,
    "LTCUSDT": 40_000_000.0,
}


@dataclass(frozen=True)
class CostScenarioEstimate:
    """Per-fill cost estimate with a named scenario."""

    policy_version: str
    scenario: str
    symbol: str
    side: str
    notional_usd: float
    adv_usd: float
    fee_usd: float
    spread_usd: float
    slippage_usd: float
    funding_usd: float
    latency_usd: float
    impact_usd: float
    total_cost_usd: float
    total_cost_bps: float

    def as_totals(self) -> dict[str, float]:
        return {
            "notional_usd": self.notional_usd,
            "fee_usd": self.fee_usd,
            "spread_usd": self.spread_usd,
            "slippage_usd": self.slippage_usd,
            "funding_usd": self.funding_usd,
            "latency_usd": self.latency_usd,
            "impact_usd": self.impact_usd,
            "total_cost_usd": self.total_cost_usd,
            "total_cost_bps": self.total_cost_bps,
        }


@dataclass(frozen=True)
class ExecutionCostPolicy:
    """Conservative cost policy for cross-stage execution reporting."""

    policy_version: str = _DEFAULT_POLICY_VERSION
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 4.0
    spread_bps: float = 1.0
    slippage_bps: float = 1.5
    funding_bps_per_8h: float = 0.0
    latency_bps_per_bar: float = 0.25
    impact_coeff: float = 10.0
    adverse_multiplier: float = 1.5
    severe_multiplier: float = 2.0
    fallback_adv_usd: dict[str, float] = field(default_factory=lambda: dict(_FALLBACK_ADV_USD))
    default_adv_usd: float = 50_000_000.0

    def _adv_usd_for_symbol(self, symbol: str, adv_usd: float | None) -> float:
        if adv_usd is not None and adv_usd > 0.0:
            return float(adv_usd)
        return float(self.fallback_adv_usd.get(symbol, self.default_adv_usd))

    def _scenario_multiplier(self, scenario: str) -> float:
        if scenario == "base":
            return 1.0
        if scenario == "adverse":
            return float(self.adverse_multiplier)
        if scenario == "severe":
            return float(self.severe_multiplier)
        raise ValueError(f"Unknown cost scenario: {scenario}")

    def estimate_fill_cost(
        self,
        symbol: str,
        side: str,
        notional_usd: float,
        *,
        adv_usd: float | None = None,
        funding_rate_bps: float = 0.0,
        latency_bars: float = 0.0,
        is_taker: bool = False,
        scenario: str = "base",
    ) -> CostScenarioEstimate:
        """Estimate one fill under a named cost scenario."""

        notional = max(float(notional_usd), 0.0)
        adv = self._adv_usd_for_symbol(symbol, adv_usd)
        if notional == 0.0:
            return CostScenarioEstimate(
                policy_version=self.policy_version,
                scenario=scenario,
                symbol=symbol,
                side=side,
                notional_usd=0.0,
                adv_usd=adv,
                fee_usd=0.0,
                spread_usd=0.0,
                slippage_usd=0.0,
                funding_usd=0.0,
                latency_usd=0.0,
                impact_usd=0.0,
                total_cost_usd=0.0,
                total_cost_bps=0.0,
            )

        mult = self._scenario_multiplier(scenario)
        fee_bps = self.taker_fee_bps if is_taker else self.maker_fee_bps
        fee_usd = notional * fee_bps / 10_000.0
        spread_usd = notional * self.spread_bps * mult / 10_000.0
        slippage_usd = notional * self.slippage_bps * mult / 10_000.0
        participation = notional / max(adv / 24.0, 1.0)
        impact_bps = self.impact_coeff * math.sqrt(participation) * mult
        impact_usd = notional * impact_bps / 10_000.0
        funding_usd = notional * abs(funding_rate_bps) / 10_000.0
        latency_usd = notional * self.latency_bps_per_bar * max(latency_bars, 0.0) * mult / 10_000.0
        total_cost_usd = fee_usd + spread_usd + slippage_usd + funding_usd + latency_usd + impact_usd
        total_cost_bps = (total_cost_usd / notional) * 10_000.0

        return CostScenarioEstimate(
            policy_version=self.policy_version,
            scenario=scenario,
            symbol=symbol,
            side=side,
            notional_usd=notional,
            adv_usd=adv,
            fee_usd=fee_usd,
            spread_usd=spread_usd,
            slippage_usd=slippage_usd,
            funding_usd=funding_usd,
            latency_usd=latency_usd,
            impact_usd=impact_usd,
            total_cost_usd=total_cost_usd,
            total_cost_bps=total_cost_bps,
        )

    def scenario_estimates(
        self,
        symbol: str,
        side: str,
        notional_usd: float,
        *,
        adv_usd: float | None = None,
        funding_rate_bps: float = 0.0,
        latency_bars: float = 0.0,
        is_taker: bool = False,
    ) -> dict[str, CostScenarioEstimate]:
        """Return base/adverse/severe cost estimates for one fill."""

        return {
            scenario: self.estimate_fill_cost(
                symbol,
                side,
                notional_usd,
                adv_usd=adv_usd,
                funding_rate_bps=funding_rate_bps,
                latency_bars=latency_bars,
                is_taker=is_taker,
                scenario=scenario,
            )
            for scenario in ("base", "adverse", "severe")
        }
