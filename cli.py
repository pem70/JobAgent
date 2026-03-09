from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import load_config, load_env
from db import get_connection
from services.fetcher import build_queries, fetch_jobs, ingest_jobs
from services.matcher import run_match_pipeline
from services.profile import get_profile, init_profile_from_resume, update_profile_fields
from services.tracker import (
    VALID_APPLICATION_STATUSES,
    append_note,
    dismiss_job,
    list_applications,
    open_job_url,
    set_job_interaction,
    upsert_application,
)

app = typer.Typer(name="agent", help="Job seeking agent CLI.")
profile_app = typer.Typer(help="Profile management commands.")
console = Console()


@app.command("health")
def health_check() -> None:
    """Verify DB and config initialization."""
    load_env()
    _ = load_config()
    conn = get_connection()
    conn.close()
    console.print("[green]OK[/green] DB and config initialized.")


@profile_app.command("init")
def profile_init() -> None:
    """Initialize profile from resume PDF and interactive prompts."""
    try:
        load_env()
        resume_path = typer.prompt("Resume PDF path").strip()
        name = typer.prompt("Name").strip()
        roles_raw = typer.prompt("Target roles (comma-separated)").strip()
        skills_raw = typer.prompt("Skills (comma-separated)").strip()
        yoe = int(typer.prompt("Years of experience", default="0").strip())
        location_pref = typer.prompt("Location preference", default="").strip()
        remote_ok = typer.confirm("Remote OK?", default=True)
        min_salary = int(typer.prompt("Minimum salary (0 if unspecified)", default="0").strip())
        deal_breakers_raw = typer.prompt("Deal breakers (comma-separated, optional)", default="").strip()

        saved, cfg = init_profile_from_resume(
            resume_path=resume_path,
            name=name,
            target_roles=[x.strip() for x in roles_raw.split(",") if x.strip()],
            skills=[x.strip() for x in skills_raw.split(",") if x.strip()],
            yoe=yoe,
            location_pref=location_pref,
            remote_ok=remote_ok,
            min_salary=min_salary,
            deal_breakers=[x.strip() for x in deal_breakers_raw.split(",") if x.strip()],
        )

        summary = (
            f"[bold]Name:[/bold] {saved['name']}\n"
            f"[bold]Target Roles:[/bold] {', '.join(saved['target_roles'])}\n"
            f"[bold]Skills:[/bold] {', '.join(saved['skills'])}\n"
            f"[bold]YoE:[/bold] {saved['yoe']}\n"
            f"[bold]Location:[/bold] {saved['location_pref']}\n"
            f"[bold]Remote OK:[/bold] {saved['remote_ok']}\n"
            f"[bold]Min Salary:[/bold] {saved['min_salary']}\n"
            f"[bold]Config must_have_roles:[/bold] {', '.join(cfg.get('must_have_roles', []))}"
        )
        console.print(Panel(summary, title="Profile Initialized", border_style="green"))
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)


@profile_app.command("show")
def profile_show() -> None:
    """Show saved profile and keyword config groups."""
    profile = get_profile()
    if not profile:
        console.print("[yellow]Run `agent profile init` first.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="User Profile")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Name", str(profile.get("name", "")))
    table.add_row("Target Roles", ", ".join(profile.get("target_roles", [])))
    table.add_row("Skills", ", ".join(profile.get("skills", [])))
    table.add_row("YoE", str(profile.get("yoe", 0)))
    table.add_row("Location Pref", str(profile.get("location_pref", "")))
    table.add_row("Remote OK", str(profile.get("remote_ok", True)))
    table.add_row("Min Salary", str(profile.get("min_salary", 0)))
    table.add_row("Deal Breakers", ", ".join(profile.get("deal_breakers", [])))
    console.print(table)

    cfg = load_config()
    cfg_table = Table(title="Keyword Config")
    cfg_table.add_column("Group", style="magenta")
    cfg_table.add_column("Values", style="white")
    cfg_table.add_row("must_have_roles", ", ".join(cfg.get("must_have_roles", [])))
    cfg_table.add_row("nice_to_have_skills", ", ".join(cfg.get("nice_to_have_skills", [])))
    cfg_table.add_row("exclude_terms", ", ".join(cfg.get("exclude_terms", [])))
    console.print(cfg_table)


@profile_app.command("update")
def profile_update(
    resume: str | None = typer.Option(None, "--resume", help="Resume PDF path."),
    skills: str | None = typer.Option(None, "--skills", help="Comma-separated skills."),
    roles: str | None = typer.Option(None, "--roles", help="Comma-separated target roles."),
    yoe: int | None = typer.Option(None, "--yoe", help="Years of experience."),
    salary: int | None = typer.Option(None, "--salary", help="Minimum salary."),
    location: str | None = typer.Option(None, "--location", help="Location preference."),
    remote: bool | None = typer.Option(None, "--remote/--no-remote", help="Remote preference."),
    deal_breakers: str | None = typer.Option(None, "--deal-breakers", help="Comma-separated deal breakers."),
) -> None:
    """Update profile fields selectively."""
    if all(
        value is None
        for value in [resume, skills, roles, yoe, salary, location, remote, deal_breakers]
    ):
        console.print("[yellow]No fields provided. Nothing to update.[/yellow]")
        raise typer.Exit(code=1)

    try:
        saved, _, changed = update_profile_fields(
            resume_path=resume,
            skills=skills,
            roles=roles,
            yoe=yoe,
            salary=salary,
            location=location,
            remote=remote,
            deal_breakers=deal_breakers,
        )
        console.print(
            Panel(
                f"Updated fields: {', '.join(sorted(changed))}\n"
                f"Name: {saved.get('name', '')}\n"
                f"Roles: {', '.join(saved.get('target_roles', []))}\n"
                f"Skills: {', '.join(saved.get('skills', []))}",
                title="Profile Updated",
                border_style="blue",
            )
        )
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)


app.add_typer(profile_app, name="profile")


@app.command("fetch")
def fetch(
    limit: int = typer.Option(50, "--limit", min=1, help="Max number of new jobs to ingest per run."),
) -> None:
    """Fetch jobs from JSearch and store filtered results."""
    load_env()
    profile = get_profile()
    if not profile:
        console.print("[yellow]Run `agent profile init` first.[/yellow]")
        raise typer.Exit(code=1)

    config = load_config()
    from config import load_synonyms

    synonyms = load_synonyms()
    queries = build_queries(config)
    if not queries:
        console.print("[yellow]No queries generated from config.json[/yellow]")
        raise typer.Exit(code=1)

    date_range = str(config.get("date_range", "3days"))
    pages = int(config.get("pages_per_query", 1))
    raw_jobs: list[dict] = []

    for query in queries:
        for page in range(1, pages + 1):
            try:
                raw_jobs.extend(fetch_jobs(query=query, date_posted=date_range, page=page))
            except Exception as exc:
                console.print(f"[red]Error fetching '{query}' page {page}: {exc}[/red]")

    conn = get_connection()
    try:
        inserted, filtered_out, duplicates = ingest_jobs(
            raw_jobs,
            config,
            synonyms,
            conn,
            limit=limit,
        )
    finally:
        conn.close()

    console.print(
        f"[green]Fetched {inserted} new jobs[/green], "
        f"[yellow]filtered out {filtered_out}[/yellow], "
        f"[cyan]{duplicates} already in database[/cyan]."
    )


@app.command("jobs")
def jobs(
    job_id: int | None = typer.Argument(None, help="Optional job id for full detail view."),
    status: str | None = typer.Option(None, "--status", help="Filter by job status."),
    company: str | None = typer.Option(None, "--company", help="Filter by company keyword."),
    remote_only: bool = typer.Option(False, "--remote-only", help="Show only remote jobs."),
) -> None:
    """List jobs or show one job detail when id is provided."""
    conn = get_connection()
    try:
        if job_id is not None:
            row = conn.execute("SELECT * FROM job_posting WHERE id = ?", (job_id,)).fetchone()
            if not row:
                console.print(f"[yellow]Job {job_id} not found.[/yellow]")
                raise typer.Exit(code=1)

            detail = Table(title=f"Job Detail #{job_id}")
            detail.add_column("Field", style="cyan")
            detail.add_column("Value", style="white")
            for key in row.keys():
                if key == "embedding":
                    continue
                val = row[key]
                if key == "parsed_skills" and val:
                    try:
                        val = ", ".join(json.loads(val))
                    except Exception:
                        pass
                detail.add_row(key, "" if val is None else str(val))
            console.print(detail)

            match_row = conn.execute(
                "SELECT final_score, rationale, skill_gaps, red_flags FROM match_result WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if match_row:
                match_table = Table(title="Match Result")
                match_table.add_column("Field", style="magenta")
                match_table.add_column("Value", style="white")
                match_table.add_row("final_score", str(match_row["final_score"]))
                match_table.add_row("rationale", str(match_row["rationale"] or ""))
                match_table.add_row("skill_gaps", str(match_row["skill_gaps"] or "[]"))
                match_table.add_row("red_flags", str(match_row["red_flags"] or "[]"))
                console.print(match_table)

            app_row = conn.execute(
                "SELECT status, notes, updated_at FROM application WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if app_row:
                app_table = Table(title="Application")
                app_table.add_column("Field", style="green")
                app_table.add_column("Value", style="white")
                app_table.add_row("status", str(app_row["status"]))
                app_table.add_row("notes", str(app_row["notes"] or ""))
                app_table.add_row("updated_at", str(app_row["updated_at"]))
                console.print(app_table)

            conn.execute("UPDATE job_posting SET interaction = 'viewed' WHERE id = ?", (job_id,))
            conn.commit()
            return

        query = """
            SELECT id, title, company, location, keyword_score, status, posted_at, remote
            FROM job_posting
            WHERE 1 = 1
        """
        params: list[object] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if company:
            query += " AND lower(company) LIKE ?"
            params.append(f"%{company.lower()}%")
        if remote_only:
            query += " AND remote = 1"
        query += " ORDER BY id DESC"

        rows = conn.execute(query, params).fetchall()
        table = Table(title="Jobs")
        table.add_column("id", justify="right")
        table.add_column("title")
        table.add_column("company")
        table.add_column("location")
        table.add_column("keyword_score", justify="right")
        table.add_column("status")
        table.add_column("posted_at")
        for row in rows:
            table.add_row(
                str(row["id"]),
                str(row["title"] or ""),
                str(row["company"] or ""),
                str(row["location"] or ""),
                f"{float(row['keyword_score'] or 0):.2f}",
                str(row["status"] or ""),
                str(row["posted_at"] or ""),
            )
        console.print(table)
    finally:
        conn.close()


@app.command("match")
def match(
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM reranking."),
    top: int = typer.Option(10, "--top", min=1, help="Number of top jobs to show."),
    include_seen: bool = typer.Option(False, "--include-seen", help="Include already seen jobs."),
    learn: bool = typer.Option(False, "--learn", help="Update learned weights before matching."),
) -> None:
    """Run 3-layer matching pipeline and display ranked recommendations."""
    load_env()
    try:
        results = run_match_pipeline(
            include_seen=include_seen,
            llm_enabled=not no_llm,
            top_n=top,
            learn_first=learn,
        )
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)

    if not results:
        console.print("[yellow]No matchable jobs found.[/yellow]")
        return

    table = Table(title="Job Recommendations")
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("Company")
    table.add_column("Score", justify="right")
    table.add_column("Rationale")
    for item in results:
        table.add_row(
            str(item.get("id", "")),
            str(item.get("title", "") or ""),
            str(item.get("company", "") or ""),
            f"{float(item.get('final_score', 0.0)):.2f}",
            str(item.get("rationale", "") or ""),
        )
    console.print(table)


@app.command("open")
def open_job(job_id: int = typer.Argument(..., help="Job id to open in browser.")) -> None:
    """Open job URL and prompt to mark application status."""
    try:
        url = open_job_url(job_id)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Opened:[/green] {url}")
    choice = typer.prompt("Mark as applied? [y/n/later]", default="later").strip().lower()
    if choice == "y":
        upsert_application(job_id, status="applied")
    elif choice == "later":
        upsert_application(job_id, status="saved")
        set_job_interaction(job_id, interaction="viewed", status="tracked")
    elif choice == "n":
        return
    else:
        console.print("[yellow]Unknown option. No changes made.[/yellow]")


@app.command("apply")
def apply_job(
    job_id: int = typer.Argument(..., help="Job id."),
    status: str = typer.Option(..., "--status", help="saved/applied/interviewing/offer/rejected"),
) -> None:
    """Create or update application status for a job."""
    normalized = status.strip().lower()
    if normalized not in VALID_APPLICATION_STATUSES:
        console.print(
            f"[yellow]Invalid status '{status}'. Use one of: {', '.join(sorted(VALID_APPLICATION_STATUSES))}[/yellow]"
        )
        raise typer.Exit(code=1)
    try:
        upsert_application(job_id, status=normalized)
        console.print(f"[green]Application updated:[/green] job={job_id} status={normalized}")
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)


@app.command("note")
def note_job(
    job_id: int = typer.Argument(..., help="Job id."),
    text: str = typer.Argument(..., help="Note text."),
) -> None:
    """Append a timestamped note to an application."""
    try:
        append_note(job_id, text)
        console.print(f"[green]Note saved for job {job_id}.[/green]")
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)


@app.command("list")
def list_apps(
    status: str | None = typer.Option(None, "--status", help="Filter by application status."),
) -> None:
    """List tracked applications."""
    rows = list_applications(status=status)
    table = Table(title="Applications")
    table.add_column("job_id", justify="right")
    table.add_column("title")
    table.add_column("company")
    table.add_column("status")
    table.add_column("updated_at")
    table.add_column("notes")
    for row in rows:
        notes = str(row.get("notes") or "")
        if len(notes) > 60:
            notes = notes[:57] + "..."
        table.add_row(
            str(row.get("job_id", "")),
            str(row.get("title", "")),
            str(row.get("company", "")),
            str(row.get("status", "")),
            str(row.get("updated_at", "")),
            notes,
        )
    console.print(table)


@app.command("dismiss")
def dismiss(job_id: int = typer.Argument(..., help="Job id to dismiss.")) -> None:
    """Dismiss a job so it will no longer appear in matching."""
    try:
        dismiss_job(job_id)
        console.print(f"[green]Job {job_id} dismissed.[/green]")
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
