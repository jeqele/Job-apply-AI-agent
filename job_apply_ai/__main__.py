"""
Main entry point for the Job Application AI Agent.

This module provides a command-line interface to run different components
of the Job Application AI Agent.
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Job Application AI Agent')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Web UI command
    web_parser = subparsers.add_parser('web', help='Start the web interface')
    web_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    web_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    # Scraper command
    scraper_parser = subparsers.add_parser('scrape', help='Scrape job listings')
    scraper_parser.add_argument('--keyword', required=True, help='Job title or keyword to search for')
    scraper_parser.add_argument('--location', required=True, help='Location to search in')
    scraper_parser.add_argument('--output', help='Output file path (Excel)')
    scraper_parser.add_argument('--max-jobs', type=int, default=10, help='Maximum number of jobs to scrape')
    scraper_parser.add_argument('--max-days-old', type=int, default=30, help='Maximum age of job postings in days')
    scraper_parser.add_argument(
        '--sources',
        default='all',
        help='Comma-separated sources: linkedin,adzuna,reed,indeed,totaljobs,cv-library,remoteok,all',
    )
    scraper_parser.add_argument(
        '--mode',
        choices=['api', 'scrape', 'both'],
        default='both',
        help='Use API, scrape, or both where available',
    )
    scraper_parser.add_argument(
        '--no-enrich',
        action='store_true',
        help='Skip fetching job details and contact emails',
    )
    
    # CV modifier command
    cv_parser = subparsers.add_parser('tailor', help='Tailor CV for a job')
    cv_parser.add_argument('--cv', required=True, help='Path to CV template (.docx)')
    cv_parser.add_argument('--job', help='Path to job description file (text)')
    cv_parser.add_argument('--jobs-file', help='Path to Excel file with multiple job listings')
    cv_parser.add_argument('--output-dir', help='Directory to save the tailored CVs')
    cv_parser.add_argument('--output', help='Output file path for single job (.docx)')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process multiple jobs and generate CVs')
    batch_parser.add_argument('--cv', required=True, help='Path to CV template (.docx)')
    batch_parser.add_argument('--jobs-file', required=True, help='Path to Excel file with job listings')
    batch_parser.add_argument('--output-dir', help='Directory to save the tailored CVs')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'web':
        # Import here to avoid circular imports
        from job_apply_ai.ui.app import app
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    elif args.command == 'scrape':
        from job_apply_ai.scraper.aggregator import search_and_save

        output_file = args.output
        if not output_file:
            output_dir = os.path.join(os.getcwd(), "job_apply_ai", "outputs", "jobs")
            os.makedirs(output_dir, exist_ok=True)
            today_date = datetime.today().strftime("%Y-%m-%d")
            output_file = os.path.join(output_dir, f"jobs_{today_date}.xlsx")

        sources = [source.strip() for source in args.sources.split(",") if source.strip()]
        jobs, filename = search_and_save(
            args.keyword,
            args.location,
            output_file=output_file,
            max_jobs=args.max_jobs,
            max_days_old=args.max_days_old,
            sources=sources,
            mode=args.mode,
            enrich_details=not args.no_enrich,
        )

        if jobs:
            logger.info(f"Saved {len(jobs)} jobs to {filename}")
            jobs_with_email = sum(1 for job in jobs if job.get("emails"))
            logger.info(f"Jobs with contact emails: {jobs_with_email}/{len(jobs)}")
        else:
            logger.warning("No jobs found")
            
    elif args.command == 'tailor':
        from job_apply_ai.cv_modifier.cv_generator import RAGCVGenerator, batch_generate_cvs
        import pandas as pd

        output_dir = args.output_dir or os.path.join(os.getcwd(), "job_apply_ai", "outputs", "cvs")
        os.makedirs(output_dir, exist_ok=True)

        if args.jobs_file:
            jobs_df = pd.read_excel(args.jobs_file)
            jobs = jobs_df.to_dict(orient="records")
            generated_cvs = batch_generate_cvs(args.cv, jobs, output_dir)
            if generated_cvs:
                logger.info(f"Generated {len(generated_cvs)} tailored CVs:")
                for cv_path in generated_cvs:
                    logger.info(f"  - {cv_path}")
            else:
                logger.warning("Failed to generate any CVs")

        elif args.job:
            try:
                with open(args.job, 'r', encoding='utf-8') as handle:
                    job_description = handle.read()
            except Exception as e:
                logger.error(f"Error reading job description: {str(e)}")
                sys.exit(1)

            job = {
                "title": "Target Role",
                "company": "Target Company",
                "description": job_description,
            }
            output_path = args.output or os.path.join(
                output_dir,
                f"Tailored_CV_{datetime.today().strftime('%Y-%m-%d')}.docx",
            )
            try:
                generator = RAGCVGenerator()
                generator.generate_cv(args.cv, job, output_path)
                logger.info(f"Tailored CV saved to {output_path}")
            except Exception as e:
                logger.error(f"Error tailoring CV: {str(e)}")
                sys.exit(1)
        else:
            logger.error("Either --job or --jobs-file must be specified")
            sys.exit(1)

    elif args.command == 'batch':
        from job_apply_ai.cv_modifier.cv_generator import batch_generate_cvs
        import pandas as pd

        output_dir = args.output_dir or os.path.join(os.getcwd(), "job_apply_ai", "outputs", "cvs")
        os.makedirs(output_dir, exist_ok=True)
        jobs_df = pd.read_excel(args.jobs_file)
        generated_cvs = batch_generate_cvs(args.cv, jobs_df.to_dict(orient="records"), output_dir)
        
        if generated_cvs:
            logger.info(f"Generated {len(generated_cvs)} tailored CVs:")
            for cv_path in generated_cvs:
                logger.info(f"  - {cv_path}")
        else:
            logger.warning("Failed to generate any CVs")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 