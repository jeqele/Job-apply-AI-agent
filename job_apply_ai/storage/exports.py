"""Export job listings to Excel, CSV, and PDF."""

import io
import logging
from datetime import datetime
from typing import BinaryIO, Literal

import pandas as pd

from job_apply_ai.job_schema import JOB_COLUMNS

logger = logging.getLogger(__name__)

ExportFormat = Literal["excel", "csv", "pdf"]

EXPORT_COLUMNS = JOB_COLUMNS + [
    "matched_skills",
    "matched_categories",
]


def jobs_to_dataframe(jobs: list[dict]) -> pd.DataFrame:
    """Convert job dicts to a normalized DataFrame for export."""
    if not jobs:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    rows = []
    for job in jobs:
        row = {column: job.get(column, "") for column in JOB_COLUMNS}
        skills = job.get("matched_skills", [])
        categories = job.get("matched_categories", {})
        row["matched_skills"] = ", ".join(skills) if isinstance(skills, list) else str(skills)
        if isinstance(categories, dict):
            row["matched_categories"] = "; ".join(
                f"{key}: {', '.join(values)}"
                for key, values in categories.items()
                if values
            )
        else:
            row["matched_categories"] = str(categories)
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered = [column for column in EXPORT_COLUMNS if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]
    return df[ordered + remaining].fillna("")


def export_jobs(
    jobs: list[dict],
    fmt: ExportFormat,
    output: str | BinaryIO | None = None,
) -> str | io.BytesIO:
    """Export jobs to the requested format."""
    df = jobs_to_dataframe(jobs)

    if fmt == "excel":
        if output is None:
            today = datetime.today().strftime("%Y-%m-%d")
            output = f"jobs_{today}.xlsx"
        df.to_excel(output, index=False)
        logger.info("Exported %s jobs to Excel", len(jobs))
        return output if isinstance(output, str) else output

    if fmt == "csv":
        if output is None:
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            return io.BytesIO(buffer.getvalue().encode("utf-8-sig"))
        if isinstance(output, str):
            df.to_csv(output, index=False)
            return output
        df.to_csv(output, index=False)
        return output

    if fmt == "pdf":
        return _export_pdf(df, jobs)

    raise ValueError(f"Unsupported export format: {fmt}")


def _export_pdf(df: pd.DataFrame, jobs: list[dict]) -> io.BytesIO:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=1 * cm,
        rightMargin=1 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    elements = [
        Paragraph(
            f"<b>Job Listings Export</b> — {datetime.today().strftime('%Y-%m-%d')}",
            styles["Title"],
        ),
        Spacer(1, 0.5 * cm),
        Paragraph(f"Total jobs: {len(jobs)}", styles["Normal"]),
        Spacer(1, 0.5 * cm),
    ]

    display_columns = [
        "title",
        "company",
        "location",
        "work_type",
        "salary",
        "source",
        "emails",
        "link",
    ]
    table_data = [[col.replace("_", " ").title() for col in display_columns]]

    for job in jobs:
        row = []
        for column in display_columns:
            value = str(job.get(column, "") or "")
            if column == "link" and len(value) > 60:
                value = value[:57] + "..."
            elif len(value) > 80:
                value = value[:77] + "..."
            row.append(value)
        table_data.append(row)

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[4.5 * cm, 3.5 * cm, 3 * cm, 2.5 * cm, 3 * cm, 2 * cm, 4 * cm, 5 * cm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    logger.info("Exported %s jobs to PDF", len(jobs))
    return buffer
