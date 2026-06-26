# Job Application AI Agent

An intelligent AI-powered tool that automates the job application process by:
1. Scraping job listings from platforms like LinkedIn
2. Analyzing job descriptions to extract key requirements
3. Automatically tailoring your CV to match job requirements
4. Generating customized cover letters

## Features

- **Job Scraping**: Automatically search and collect job listings from LinkedIn
- **Intelligent Analysis**: Extract key skills and requirements from job descriptions
- **CV Customization**: Tailor your CV to highlight relevant skills for each job
- **Batch Processing**: Generate multiple tailored CVs for different jobs at once
- **User-Friendly Interface**: Simple web interface to control the entire process

## Setup

### Prerequisites

- Python 3.8+
- Chrome browser (for web scraping)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Job-apply-AI-agent.git
cd Job-apply-AI-agent
```

2. Run the installation script:
```bash
# On Unix-based systems (macOS, Linux)
./install.sh

# On Windows
install.bat
```

This will:
- Create a virtual environment
- Install all dependencies
- Download the required spaCy language model
- Install the package in development mode

## Usage

### Web Interface

1. Start the web interface:
```bash
# Activate the virtual environment first
source venv/bin/activate  # On Unix-based systems
venv\Scripts\activate.bat  # On Windows

# Start the web app
job-apply-ai web
```

2. Open your browser and go to: http://localhost:5000

3. Upload your base CV template

4. Search for jobs by entering a job title and location, or use **Batch search** (Step 2 → **Batch search** tab) to upload title and location files.

5. Generate tailored CVs for all jobs or for specific jobs

### Batch job search

Search every job title in every location by uploading two plain-text files (one entry per line). HermesHire builds a queue of all combinations, runs each search, deduplicates results, and saves them to the database (web) or Excel (CLI).

**Web:** Home → Step 2 — Search for jobs → **Batch search** tab (or use the **Batch search** quick link). Upload `titles.txt` and `locations.txt`, or paste lines into the text areas. You can use a file or pasted text for each side — at least one source per column is required.

**Example files:** see `examples/batch_search/titles.txt` and `examples/batch_search/locations.txt`.

**titles.txt**
```
Software Engineer
Data Scientist
# lines starting with # are ignored
Product Manager
```

**locations.txt**
```
Berlin
Remote
London
```

With 3 titles and 3 locations, HermesHire queues **9 searches** (every title × every location). Default maximum is **100 combinations** per batch; set `MAX_BATCH_SEARCH_COMBINATIONS` in `.env` to raise it (see `.env.example`).

**CLI:**

```bash
python -m job_apply_ai batch-search \
  --titles-file examples/batch_search/titles.txt \
  --locations-file examples/batch_search/locations.txt \
  --max-jobs 10
```

Windows (PowerShell):

```powershell
python -m job_apply_ai batch-search `
  --titles-file examples/batch_search/titles.txt `
  --locations-file examples/batch_search/locations.txt `
  --max-jobs 10
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-jobs` | `10` | Max jobs fetched per title/location pair |
| `--sources` | `all` | Comma-separated sources (`linkedin-mcp,linkedin,adzuna,reed,indeed`, or `all`) |
| `--mode` | `both` | `api`, `scrape`, or `both` |
| `--output` | auto | Excel output path (defaults to `job_apply_ai/outputs/jobs/batch_jobs_YYYY-MM-DD.xlsx`) |
| `--no-enrich` | off | Skip fetching full job details and contact emails |

### LinkedIn MCP job search

HermesHire can search LinkedIn through a local [linkedin-mcp-server](https://github.com/stickerdaniel/linkedin-mcp-server) sidecar using your logged-in browser session. This is more reliable than anonymous Selenium scraping and is the recommended LinkedIn source (`linkedin-mcp`).

**Setup (one time):**

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Log in to LinkedIn in the MCP browser profile:
   ```bash
   uvx mcp-server-linkedin@latest --login
   ```

**Start the sidecar** (keep this running while searching):

```powershell
# Windows
.\scripts\start-linkedin-mcp.ps1
```

```bash
# macOS / Linux
./scripts/start-linkedin-mcp.sh
```

**Use in searches:** include `linkedin-mcp` in job sources (it is the default in the web UI). Configure the sidecar URL in `.env` if needed (`LINKEDIN_MCP_URL=http://127.0.0.1:8765/mcp`).

**Port conflict on Windows:** Docker often binds port 8080. The start scripts default to **8765** instead. If you change the port, set both `LINKEDIN_MCP_PORT` when starting the sidecar and `LINKEDIN_MCP_URL` in `.env`.

**Rate limits and account safety:** LinkedIn does not publish a safe requests-per-hour limit. The MCP server queues tool calls sequentially, but batch searches still issue many requests. Use conservative settings when `linkedin-mcp` is enabled:

- `LINKEDIN_MCP_BATCH_DELAY_SECONDS=30` — pause between batch combinations (default when LinkedIn sources are used)
- `LINKEDIN_MCP_MAX_PAGES=1` — limit search result pages per query
- `LINKEDIN_MCP_DETAIL_DELAY_SECONDS=2` — pause between job detail fetches
- Keep batch matrices small; prefer API sources (Adzuna, Reed) for large title × location grids
- Do not run `linkedin` (Selenium) and `linkedin-mcp` together in the same batch

See the maintainer guidance in [discussion #351](https://github.com/stickerdaniel/linkedin-mcp-server/discussions/351).

### Command Line

The application also provides a command-line interface:

```bash
# Scrape job listings
job-apply-ai scrape --keyword "Software Engineer" --location "Berlin" --max-jobs 5

# Batch search: every title in every location (see Batch job search above)
python -m job_apply_ai batch-search \
  --titles-file examples/batch_search/titles.txt \
  --locations-file examples/batch_search/locations.txt \
  --max-jobs 10

# Generate tailored CVs for all jobs in an Excel file
job-apply-ai batch --cv path/to/cv_template.docx --jobs-file path/to/jobs.xlsx

# Generate a tailored CV for a single job description
job-apply-ai tailor --cv path/to/cv_template.docx --job path/to/job_description.txt
```

## Project Structure

- `job_apply_ai/scraper/`: Job listing scraping modules
- `job_apply_ai/cv_modifier/`: CV customization functionality
- `job_apply_ai/utils/`: Utility functions and helpers
- `job_apply_ai/ui/`: User interface components
- `job_apply_ai/outputs/`: Output directories for jobs and CVs
  - `job_apply_ai/outputs/jobs/`: Contains Excel files with job listings
  - `job_apply_ai/outputs/cvs/`: Contains generated CV files

## Testing

For detailed testing instructions, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
