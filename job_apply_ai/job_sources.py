"""Job source identifiers shared across UI, storage, and scraping."""

UI_JOB_SOURCE_OPTIONS: tuple[str, ...] = (
    "linkedin-mcp",
    "linkedin",
    "adzuna",
    "reed",
    "indeed",
    "totaljobs",
    "cv-library",
    "remoteok",
    "arbeitnow",
)

UI_JOB_SOURCE_LABELS: dict[str, str] = {
    "linkedin-mcp": "LinkedIn MCP",
    "linkedin": "LinkedIn",
    "adzuna": "Adzuna",
    "reed": "Reed",
    "indeed": "Indeed",
    "totaljobs": "Totaljobs",
    "cv-library": "CV Library",
    "remoteok": "RemoteOK",
    "arbeitnow": "Arbeitnow",
}

UI_JOB_SOURCE_DESCRIPTIONS: dict[str, str] = {
    "linkedin-mcp": "LinkedIn jobs via MCP integration",
    "linkedin": "LinkedIn job listings",
    "arbeitnow": "Europe-focused job API",
    "remoteok": "Remote jobs API",
}

UI_DEFAULT_JOB_SOURCES = ",".join(UI_JOB_SOURCE_OPTIONS)
UI_JOB_SOURCES_PLACEHOLDER = UI_DEFAULT_JOB_SOURCES


def parse_sources_csv(sources: str | None) -> list[str]:
    """Split a comma-separated sources string into normalized ids."""
    if not sources:
        return []
    return [source.strip() for source in sources.split(",") if source.strip()]


def format_sources_csv(sources: list[str]) -> str:
    """Join source ids into the comma-separated storage format."""
    return ",".join(source.strip() for source in sources if source.strip())


def selected_source_ids_from_csv(sources: str | None) -> set[str]:
    """Return source ids that should appear checked in the UI."""
    parsed = parse_sources_csv(sources)
    if not parsed:
        return set(UI_JOB_SOURCE_OPTIONS)
    if "all" in parsed:
        return set(UI_JOB_SOURCE_OPTIONS)
    known = {source for source in parsed if source in UI_JOB_SOURCE_OPTIONS}
    return known or set(UI_JOB_SOURCE_OPTIONS)


def job_source_options_for_ui() -> list[dict[str, str]]:
    """Source metadata for checkbox rendering."""
    options = []
    for source_id in UI_JOB_SOURCE_OPTIONS:
        options.append(
            {
                "id": source_id,
                "label": UI_JOB_SOURCE_LABELS.get(source_id, source_id),
                "description": UI_JOB_SOURCE_DESCRIPTIONS.get(source_id, ""),
            }
        )
    return options
