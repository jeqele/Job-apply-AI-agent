"""Indeed UK job source (scrape-based)."""

import logging
import time
from datetime import datetime
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.browser import create_chrome_driver
from job_apply_ai.scraper.email_extractor import enrich_job_emails
from job_apply_ai.scraper.job_metadata import extract_salary, infer_work_type

logger = logging.getLogger(__name__)


class IndeedJobSource(JobSource):
    source_name = "Indeed"
    supports_api = False
    supports_scrape = True

    BASE_URL = "https://uk.indeed.com/jobs"

    def fetch_via_api(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        return []

    def fetch_via_scrape(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        search_url = (
            f"{self.BASE_URL}?q={quote_plus(keyword)}&l={quote_plus(location)}"
            f"&fromage={max_days_old}"
        )
        driver = create_chrome_driver(headless=self.headless)
        jobs = []

        try:
            driver.get(search_url)
            for _ in range(3):
                driver.execute_script("window.scrollBy(0, 900);")
                time.sleep(1.5)

            wait = WebDriverWait(driver, 15)
            try:
                wait.until(EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    ".job_seen_beacon, .jobsearch-SerpJobCard, div[data-jk]",
                )))
            except TimeoutException:
                logger.warning("Indeed returned no job cards")
                return []

            cards = driver.find_elements(
                By.CSS_SELECTOR,
                ".job_seen_beacon, .jobsearch-SerpJobCard, div[data-jk]",
            )
            for card in cards[:max_jobs]:
                try:
                    title_elem = card.find_element(
                        By.CSS_SELECTOR,
                        "h2.jobTitle a, h2 a, .jcs-JobTitle",
                    )
                    title = title_elem.text.strip()
                    link = title_elem.get_attribute("href") or ""
                except Exception:
                    continue

                company = self._safe_text(card, ".companyName, [data-testid='company-name']")
                location_text = self._safe_text(
                    card, ".companyLocation, [data-testid='text-location']"
                )
                salary = self._safe_text(card, ".salary-snippet, .salaryOnly, .metadata.salary")

                job = {
                    "title": title,
                    "company": company,
                    "location": location_text or location,
                    "salary": salary,
                    "work_type": infer_work_type(title, location_text or location),
                    "link": link,
                }
                enrich_job_emails(job, html=card.get_attribute("outerHTML") or "", fetch_page=False)
                jobs.append(job)
        finally:
            driver.quit()
        return jobs

    @staticmethod
    def _safe_text(parent, selector: str) -> str:
        try:
            return parent.find_element(By.CSS_SELECTOR, selector).text.strip()
        except Exception:
            return ""

    def fetch_job_details_batch(self, jobs: list[dict]) -> list[dict]:
        driver = create_chrome_driver(headless=self.headless)
        try:
            for job in jobs:
                if not job.get("link"):
                    continue
                try:
                    driver.get(job["link"])
                    time.sleep(1.5)
                    html = driver.page_source
                    soup = BeautifulSoup(html, "html.parser")
                    description_elem = soup.select_one(
                        "#jobDescriptionText, .jobsearch-JobComponent-description"
                    )
                    if description_elem:
                        job["description"] = description_elem.get_text("\n", strip=True)
                    if not job.get("salary"):
                        salary_elem = soup.select_one(".jobsearch-JobMetadataHeader-item")
                        if salary_elem:
                            job["salary"] = salary_elem.get_text(" ", strip=True)
                    enrich_job_emails(job, html=html, fetch_page=False)
                except Exception as exc:
                    logger.debug("Indeed detail fetch failed for %s: %s", job.get("link"), exc)
        finally:
            driver.quit()
        return jobs
