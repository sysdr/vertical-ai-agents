"""
TrafficSplitter — simplified from L73.
Manages variant weights; called by DriftResponder on rollback.
"""
from dataclasses import dataclass


@dataclass
class VariantConfig:
    name: str
    weight: int = 100


class TrafficSplitter:
    def __init__(self) -> None:
        self.variants: dict[str, VariantConfig] = {
            "stable": VariantConfig("stable", 100),
            "canary": VariantConfig("canary", 0),
        }

    async def rebalance(self, target_variant: str = "stable") -> None:
        """Collapse all traffic to target_variant (rollback pattern)."""
        for name in self.variants:
            self.variants[name].weight = 100 if name == target_variant else 0

    def get_weights(self) -> dict[str, int]:
        return {k: v.weight for k, v in self.variants.items()}

    def route(self) -> str:
        """Weighted random variant selection."""
        import random
        total = sum(v.weight for v in self.variants.values())
        if total == 0:
            return "stable"
        r = random.randint(0, total - 1)
        cumulative = 0
        for name, config in self.variants.items():
            cumulative += config.weight
            if r < cumulative:
                return name
        return "stable"
