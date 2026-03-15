"""
Data Source Registry — modular plugin system for data pipelines.

Add a new data source by:
1. Subclass DataSource
2. Implement fetch() → str
3. Register with @register_source or call registry.register()

Sources auto-discover API keys from env vars and can be enabled/disabled
via POLYSWARM_SOURCES env var (comma-separated whitelist).

Example:
    @register_source
    class MySource(DataSource):
        name = "my_source"
        category = "sentiment"
        description = "My custom data source"
        requires_key = "MY_API_KEY"  # optional

        def fetch(self) -> str:
            resp = self.http.get("https://api.example.com/data")
            return f"My data: {resp.json()}"
"""

from __future__ import annotations
import os
import httpx
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable


# ═══════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════

class DataSource:
    """Base class for all data sources."""

    # Override these in subclasses
    name: str = ""                    # unique identifier
    category: str = "general"         # grouping: market, derivatives, onchain, defi, sentiment, social, prediction_markets
    description: str = ""             # human-readable description
    requires_key: str | None = None   # env var name for API key (None = no key needed)
    priority: int = 50                # 0-100, higher = fetched first in output ordering
    timeout: int = 10                 # request timeout in seconds
    enabled: bool = True              # can be disabled

    def __init__(self):
        self.http = httpx.Client(timeout=self.timeout)
        self._api_key: str | None = None
        if self.requires_key:
            self._api_key = os.getenv(self.requires_key)

    @property
    def api_key(self) -> str | None:
        """Get the API key from environment."""
        if self.requires_key:
            return os.getenv(self.requires_key)
        return None

    @property
    def has_key(self) -> bool:
        """Check if the required API key is available."""
        if not self.requires_key:
            return True
        return bool(os.getenv(self.requires_key))

    @property
    def available(self) -> bool:
        """Source is available if enabled and has required API key."""
        return self.enabled and self.has_key

    def fetch(self) -> str:
        """Fetch data and return formatted string. Override in subclass."""
        raise NotImplementedError

    def safe_fetch(self) -> str:
        """Fetch with error handling — never raises."""
        try:
            if not self.available:
                return ""
            return self.fetch()
        except Exception:
            return ""

    def __repr__(self):
        status = "ready" if self.available else "needs_key" if not self.has_key else "disabled"
        return f"<{self.name} [{self.category}] {status}>"


# ═══════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════

class SourceRegistry:
    """Central registry for all data sources."""

    def __init__(self):
        self._sources: dict[str, DataSource] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, source_class: type[DataSource]) -> type[DataSource]:
        """Register a data source class."""
        instance = source_class()
        self._sources[instance.name] = instance
        cat = instance.category
        if cat not in self._categories:
            self._categories[cat] = []
        self._categories[cat].append(instance.name)
        return source_class

    def get(self, name: str) -> DataSource | None:
        return self._sources.get(name)

    def all(self) -> list[DataSource]:
        return list(self._sources.values())

    def available(self) -> list[DataSource]:
        """Return only sources that are ready to fetch."""
        # check whitelist env var
        whitelist = os.getenv("POLYSWARM_SOURCES")
        if whitelist:
            allowed = {s.strip() for s in whitelist.split(",")}
            return [s for s in self._sources.values() if s.name in allowed and s.available]
        return [s for s in self._sources.values() if s.available]

    def by_category(self, category: str) -> list[DataSource]:
        names = self._categories.get(category, [])
        return [self._sources[n] for n in names if n in self._sources]

    @property
    def categories(self) -> list[str]:
        return list(self._categories.keys())

    def status(self) -> list[dict]:
        """Return status of all registered sources."""
        return [
            {
                "name": s.name,
                "category": s.category,
                "description": s.description,
                "enabled": s.enabled,
                "requires_key": s.requires_key,
                "has_key": s.has_key,
                "available": s.available,
                "priority": s.priority,
            }
            for s in sorted(self._sources.values(), key=lambda x: (-x.priority, x.category, x.name))
        ]

    def fetch_all(self, max_workers: int = 12, timeout: int = 15) -> dict[str, str]:
        """Fetch all available sources in parallel."""
        sources = self.available()
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_source = {
                executor.submit(s.safe_fetch): s for s in sources
            }
            for future in as_completed(future_to_source, timeout=timeout):
                source = future_to_source[future]
                try:
                    result = future.result(timeout=source.timeout)
                    if result:
                        results[source.name] = result
                except Exception:
                    pass

        return results

    def build_context(self, question: str = "") -> str:
        """Build full context string from all sources, ordered by priority."""
        sections = [f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"]

        results = self.fetch_all()

        # sort by source priority (higher first)
        available = self.available()
        priority_order = sorted(available, key=lambda s: -s.priority)

        for source in priority_order:
            if source.name in results:
                sections.append(results[source.name])

        # question-specific search (prediction markets)
        if question:
            for source in available:
                if hasattr(source, 'search') and callable(source.search):
                    try:
                        search_result = source.search(question)
                        if search_result:
                            sections.append(search_result)
                    except Exception:
                        pass

        return "\n\n".join(sections)


# Global registry instance
registry = SourceRegistry()


def register_source(cls: type[DataSource]) -> type[DataSource]:
    """Decorator to register a data source."""
    registry.register(cls)
    return cls
