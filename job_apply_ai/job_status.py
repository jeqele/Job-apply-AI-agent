"""Job workflow statuses for application tracking."""

DEFAULT_JOB_STATUS = "new"

# Ordered for sidebar display (pipeline flow, then terminal states).
JOB_WORKFLOW_STATUSES = [
    "new",
    "shortlisted",
    "applying",
    "applied",
    "cv_sent",
    "interview_scheduled",
    "interview_session_approve",
    "interview_completed",
    "offer_received",
    "offer_accepted",
    "offer_declined",
    "rejected",
    "denied",
    "not_invited",
    "withdrawn",
    "on_hold",
    "archived",
]

JOB_STATUS_LABELS = {
    "new": "New / Discovered",
    "shortlisted": "Shortlisted",
    "applying": "Applying",
    "applied": "Applied",
    "cv_sent": "CV Sent",
    "interview_scheduled": "Interview Scheduled",
    "interview_session_approve": "Interview Approved",
    "interview_completed": "Interview Completed",
    "offer_received": "Offer Received",
    "offer_accepted": "Offer Accepted",
    "offer_declined": "Offer Declined",
    "rejected": "Rejected by Employer",
    "denied": "Denied",
    "not_invited": "Not Invited",
    "withdrawn": "Withdrawn",
    "on_hold": "On Hold",
    "archived": "Archived",
}

JOB_STATUS_ICONS = {
    "new": "fa-inbox",
    "shortlisted": "fa-star",
    "applying": "fa-pen",
    "applied": "fa-paper-plane",
    "cv_sent": "fa-file-alt",
    "interview_scheduled": "fa-calendar-check",
    "interview_session_approve": "fa-thumbs-up",
    "interview_completed": "fa-comments",
    "offer_received": "fa-gift",
    "offer_accepted": "fa-check-circle",
    "offer_declined": "fa-times-circle",
    "rejected": "fa-ban",
    "denied": "fa-hand-paper",
    "not_invited": "fa-user-slash",
    "withdrawn": "fa-undo",
    "on_hold": "fa-pause-circle",
    "archived": "fa-archive",
}

JOB_STATUS_BADGE_CLASSES = {
    "new": "bg-primary",
    "shortlisted": "bg-info",
    "applying": "bg-warning text-dark",
    "applied": "bg-secondary",
    "cv_sent": "bg-info",
    "interview_scheduled": "bg-primary",
    "interview_session_approve": "bg-success",
    "interview_completed": "bg-primary",
    "offer_received": "bg-success",
    "offer_accepted": "bg-success",
    "offer_declined": "bg-secondary",
    "rejected": "bg-danger",
    "denied": "bg-danger",
    "not_invited": "bg-secondary",
    "withdrawn": "bg-secondary",
    "on_hold": "bg-warning text-dark",
    "archived": "bg-dark",
}


def is_valid_job_status(status: str | None) -> bool:
    return status in JOB_WORKFLOW_STATUSES


def job_status_label(status: str | None) -> str:
    if not status:
        return JOB_STATUS_LABELS[DEFAULT_JOB_STATUS]
    return JOB_STATUS_LABELS.get(status, status.replace("_", " ").title())
