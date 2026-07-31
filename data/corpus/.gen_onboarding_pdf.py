from __future__ import annotations

from pathlib import Path
import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas


def _wrap_text(text: str, width: int = 90) -> list[str]:
    return textwrap.wrap(text, width=width, replace_whitespace=False)


def _draw_text_page(canvas: Canvas, title: str, body: str) -> None:
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(72, 760, title)
    canvas.setFont("Helvetica", 11)
    y = 730

    for paragraph in body.strip().split("\n\n"):
        if paragraph.startswith("## "):
            lines = paragraph.splitlines()
            heading = lines[0][3:].strip()
            canvas.setFont("Helvetica-Bold", 13)
            canvas.drawString(72, y, heading)
            y -= 20
            canvas.setFont("Helvetica", 11)
            remainder = "\n".join(lines[1:]).strip()
            if remainder:
                for line in remainder.split("\n"):
                    if line.startswith("- "):
                        wrapped = _wrap_text(line[2:], width=80)
                        for wrapped_line in wrapped:
                            canvas.drawString(90, y, f"- {wrapped_line}")
                            y -= 16
                    else:
                        wrapped = _wrap_text(line, width=90)
                        for wrapped_line in wrapped:
                            canvas.drawString(72, y, wrapped_line)
                            y -= 16
                    y -= 4
            y -= 6
            continue

        for line in paragraph.split("\n"):
            if line.startswith("- "):
                wrapped = _wrap_text(line[2:], width=80)
                for wrapped_line in wrapped:
                    canvas.drawString(90, y, f"- {wrapped_line}")
                    y -= 16
            else:
                wrapped = _wrap_text(line, width=90)
                for wrapped_line in wrapped:
                    canvas.drawString(72, y, wrapped_line)
                    y -= 16
            y -= 4

        y -= 6

    canvas.showPage()


def generate_pdf(path: Path) -> None:
    pages = [
        (
            "Acme Cloud Storage — New Engineer Onboarding Handbook",
            "Welcome to the Storage Platform team. This handbook covers what you need "
            "during your first two weeks: account setup, local development, and how the "
            "on-call rotation works. Read it alongside the architecture overview document, "
            "which describes how the system fits together.\n\n"
            "## Week one: accounts and access\n"
            "On your first day, your manager will sponsor your access requests for the internal "
            "developer portal, the source code repositories, and the staging cluster. Most access "
            "grants complete within a few hours, but cluster access can take up to one business day "
            "because it requires a security review. While you wait, set up your laptop using the "
            "bootstrap script in the team repository's tools directory.\n\n"
            "You will also be issued a personal API key for the staging environment. Treat it like a "
            "password: do not commit it to source control, and rotate it immediately if you suspect "
            "it has leaked. Production API keys are issued separately and require manager approval."
        ),
        (
            "Local development and review norms",
            "Clone the monorepo and run `make bootstrap` to install toolchain dependencies and "
            "pre-commit hooks. The storage platform services run locally via the dev-cluster "
            "docker-compose file in infra/dev-cluster. Bring it up with `make dev-up` and tear it down "
            "with `make dev-down`.\n\n"
            "Integration tests against the dev cluster live in the tests/integration directory and "
            "take about ten minutes to run. Run the fast unit suite with `make test` before every push; "
            "the integration suite runs automatically in CI on each pull request.\n\n"
            "## Code review norms\n"
            "Every change needs at least one approval from a team member other than the author. "
            "Storage-shard and replication code additionally requires sign-off from a senior engineer, "
            "since bugs there can affect data durability. Keep pull requests small — large, multi-purpose "
            "PRs take longer to review and are more likely to introduce regressions."
        ),
        (
            "On-call rotation and support",
            "New engineers join the on-call rotation after their first month, shadowing an experienced "
            "on-call engineer for one full rotation before taking a shift solo. Rotations are weekly and "
            "handoffs happen on Tuesdays at 10:00 in the team's local time zone.\n\n"
            "If you are paged, acknowledge the alert within five minutes. Use the runbooks linked from "
            "each alert as your starting point — most incidents map to a documented procedure. If you are "
            "unsure how to proceed, escalate early by paging the secondary on-call rather than spending too long "
            "investigating alone.\n\n"
            "## Where to ask questions\n"
            "Use the #storage-platform channel for general questions and #acs-oncall for anything urgent and "
            "production-affecting. There are no bad questions during your first few months — asking early is much "
            "cheaper than debugging a larger problem later."
        ),
        (
            "What to read first",
            "Start by reading the architecture overview, the API reference, and the onboarding runbook. "
            "Then review the storage shard service README and the shard manager design doc in docs/. "
            "Focus on the traffic flow: Gateway -> Shard Manager -> Storage Shard -> Metadata Store.\n\n"
            "Key repos and services to know:\n"
            "- `gateway-service`: request ingress, auth, rate limiting, and request routing.\n"
            "- `shard-manager`: shard ownership, consistent hashing, and rebalancing.\n"
            "- `storage-shard`: object storage, replication, and local disk durability.\n"
            "- `metadata-store`: object metadata, listings, access control, and policy enforcement.\n\n"
            "Understanding the separation between object bytes and metadata is critical: object bytes are stored on the "
            "shard cluster, while the metadata store is the source of truth for routing and permissions."
        ),
        (
            "First tasks and learning goals",
            "Your first week should include:\n"
            "- verifying your laptop bootstrap and repo access,\n"
            "- running the local dev cluster and a simple end-to-end test,\n"
            "- reading the onboarding docs, architecture overview, and API reference,\n"
            "- shadowing a code review and asking two clarifying questions about the current design.\n\n"
            "During week two, try to contribute a small documentation fix or a low-risk reliability improvement. "
            "This helps you learn the codebase and the release process while giving you an early feedback cycle.\n\n"
            "## Team culture\n"
            "We value clear communication, incremental improvements, and operational ownership. If you see a "
            "repeatedly confusing workflow, suggest a doc update or a tooling improvement rather than letting the pain persist."
        ),
    ]

    canvas = Canvas(str(path), pagesize=letter)

    for title, body in pages:
        _draw_text_page(canvas, title, body)

    canvas.save()


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent / "onboarding_handbook.pdf"
    generate_pdf(output_path)
    print(f"Wrote {output_path}")
