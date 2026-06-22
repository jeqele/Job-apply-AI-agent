"""
LinkedIn Job Scraper Module

This module provides functionality to scrape job listings from LinkedIn,
including job titles, company names, links, and full job descriptions.
"""

import os
import time
import logging
from datetime import datetime, timedelta

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from job_apply_ai.scraper.browser import create_chrome_driver
from job_apply_ai.scraper.email_extractor import enrich_job_emails
from job_apply_ai.scraper.job_metadata import (
    empty_job_details,
    extract_relocation_info,
    extract_salary,
    infer_work_type,
    parse_relocation_support,
    parse_visa_sponsorship,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LinkedInScraper:
    """
    A class to scrape job listings from LinkedIn.
    """
    
    def __init__(self, headless=True):
        """
        Initialize the LinkedIn scraper.
        
        Args:
            headless (bool): Whether to run the browser in headless mode.
        """
        self.headless = headless
        self._driver = None

    @staticmethod
    def _element_text(element):
        """Return visible or screen-reader text from a LinkedIn card element."""
        if element is None:
            return ""
        text = (element.get_attribute("textContent") or element.text or "").strip()
        return " ".join(text.split())

    @staticmethod
    def _find_elements_text(driver, selector):
        return [
            LinkedInScraper._element_text(element)
            for element in driver.find_elements(By.CSS_SELECTOR, selector)
            if LinkedInScraper._element_text(element)
        ]

    @staticmethod
    def _find_element_text(driver, selector):
        try:
            return LinkedInScraper._element_text(driver.find_element(By.CSS_SELECTOR, selector))
        except NoSuchElementException:
            return ""

    @staticmethod
    def _extract_criteria(driver):
        criteria = {}
        items = driver.find_elements(By.CSS_SELECTOR, ".description__job-criteria-item")
        for item in items:
            try:
                label = LinkedInScraper._element_text(
                    item.find_element(By.CSS_SELECTOR, ".description__job-criteria-subheader")
                ).lower()
                value = LinkedInScraper._element_text(
                    item.find_element(By.CSS_SELECTOR, ".description__job-criteria-text")
                )
                if "seniority" in label:
                    criteria["seniority_level"] = value
                elif "employment" in label:
                    criteria["employment_type"] = value
                elif "function" in label:
                    criteria["job_function"] = value
                elif "industr" in label:
                    criteria["industry"] = value
            except NoSuchElementException:
                continue
        return criteria
        
    def _configure_driver(self):
        """Configure and return a Chrome WebDriver."""
        return create_chrome_driver(headless=self.headless)
    
    def scrape_job_listings(
        self,
        keyword,
        location,
        max_jobs=10,
        max_days_old=14,
        search_filters=None,
    ):
        """
        Scrape job listings from LinkedIn based on keyword and location.
        
        Args:
            keyword (str): Job title or keyword to search for.
            location (str): Location to search in.
            max_jobs (int): Maximum number of jobs to scrape.
            max_days_old (int): Maximum age of job postings in days.
            
        Returns:
            list: List of dictionaries containing job details.
        """
        logger.info(f"Scraping LinkedIn jobs for '{keyword}' in '{location}'")
        
        driver = self._configure_driver()
        search_url = (
            f"https://www.linkedin.com/jobs/search?"
            f"keywords={keyword.replace(' ', '%20')}&location={location.replace(' ', '%20')}"
        )
        if search_filters and getattr(search_filters, "remote", False):
            search_url += "&f_WT=2"
        
        try:
            driver.get(search_url)
            
            # Scroll to load more jobs
            for _ in range(3):
                driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(2)
            
            # Wait for job listings to appear
            wait = WebDriverWait(driver, 15)
            try:
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "base-card")))
            except TimeoutException:
                logger.warning("No job listings found")
                driver.quit()
                return []
            
            jobs = []
            today = datetime.today()
            job_elements = driver.find_elements(By.CLASS_NAME, "base-card")
            
            for job in job_elements[:max_jobs]:
                try:
                    title_elem = job.find_element(By.CSS_SELECTOR, "h3.base-search-card__title, h3")
                    title = self._element_text(title_elem)

                    company = ""
                    try:
                        company_elem = job.find_element(
                            By.CSS_SELECTOR, "h4.base-search-card__subtitle a, h4.base-search-card__subtitle"
                        )
                        company = self._element_text(company_elem)
                    except NoSuchElementException:
                        pass

                    link_elem = job.find_element(By.CSS_SELECTOR, "a.base-card__full-link, a")
                    link = link_elem.get_attribute("href")

                    location_text = ""
                    try:
                        location_elem = job.find_element(By.CSS_SELECTOR, ".job-search-card__location")
                        location_text = self._element_text(location_elem)
                    except NoSuchElementException:
                        pass

                    listing_benefit = ""
                    try:
                        benefit_elem = job.find_element(By.CSS_SELECTOR, ".job-posting-benefits__text")
                        listing_benefit = self._element_text(benefit_elem)
                    except NoSuchElementException:
                        pass

                    posted_date = ""
                    try:
                        date_element = job.find_element(By.CSS_SELECTOR, "time")
                        posted_time = date_element.get_attribute("datetime")
                        if posted_time:
                            posted_date = posted_time[:10]
                            posted_date_obj = datetime.strptime(posted_date, "%Y-%m-%d")
                            days_ago = (today - posted_date_obj).days
                            if days_ago > max_days_old:
                                logger.info(f"Skipping job: {title} (Posted {days_ago} days ago)")
                                continue
                        else:
                            days_ago = "Unknown"
                    except NoSuchElementException:
                        logger.warning(f"Could not find post time for: {title}, assuming it's recent")
                        days_ago = "Unknown"
                    
                    jobs.append({
                        **empty_job_details(),
                        "title": title,
                        "company": company,
                        "location": location_text,
                        "work_type": infer_work_type(title, location_text),
                        "listing_benefit": listing_benefit,
                        "posted_date": posted_date,
                        "link": link,
                        "source": "LinkedIn",
                        "posted_days_ago": days_ago,
                        "fetch_method": "scrape",
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing job listing: {str(e)}")
                    continue
            
            logger.info(f"Successfully scraped {len(jobs)} job listings")
            return jobs
            
        except Exception as e:
            logger.error(f"Error during job scraping: {str(e)}")
            return []
            
        finally:
            driver.quit()
    
    def fetch_job_details(self, job_url, driver=None, fallback_title="", fallback_location=""):
        """
        Fetch enriched job details from a LinkedIn job URL.

        Returns:
            dict: Parsed job detail fields.
        """
        logger.info(f"Fetching job details from {job_url}")

        details = empty_job_details()
        owns_driver = driver is None
        if owns_driver:
            driver = self._configure_driver()

        try:
            driver.get(job_url)
            wait = WebDriverWait(driver, 15)

            try:
                title_elem = wait.until(EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    "h1.topcard__title, h1.top-card-layout__title, h2.top-card-layout__title"
                )))
                details["title"] = self._element_text(title_elem)
            except TimeoutException:
                details["title"] = fallback_title

            details["company"] = self._find_element_text(
                driver,
                "a.topcard__org-name-link, span.topcard__flavor--black-link"
            )
            details["company_url"] = ""
            try:
                company_link = driver.find_element(
                    By.CSS_SELECTOR, "a.topcard__org-name-link"
                ).get_attribute("href")
                details["company_url"] = company_link or ""
            except NoSuchElementException:
                pass

            location = self._find_element_text(driver, "span.topcard__flavor--bullet")
            if location:
                details["location"] = location

            details["posted_date"] = self._find_element_text(
                driver, "span.posted-time-ago__text, time"
            )
            details["applicant_count"] = self._find_element_text(
                driver, ".num-applicants__caption"
            )

            details["salary"] = self._find_element_text(
                driver, ".salary.compensation__salary, .compensation__salary"
            )

            criteria = self._extract_criteria(driver)
            details.update({key: value for key, value in criteria.items() if value})

            try:
                desc_elem = wait.until(EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    ".description__text, .show-more-less-html__markup, .decorated-job-posting__details"
                )))
                details["description"] = self._element_text(desc_elem)
            except TimeoutException:
                details["description"] = ""

            page_text = self._element_text(driver.find_element(By.TAG_NAME, "body"))
            if not details["salary"]:
                details["salary"] = extract_salary(page_text, details["description"])

            details["work_type"] = infer_work_type(
                details.get("title") or fallback_title,
                details.get("location") or fallback_location,
                details["description"],
            )
            details["visa_sponsorship"] = parse_visa_sponsorship(details["description"])
            details["relocation_support"] = parse_relocation_support(details["description"])
            details["relocation_info"] = extract_relocation_info(details["description"])
            enrich_job_emails(details, html=driver.page_source, fetch_page=False)

            return details

        except Exception as e:
            logger.error(f"Error fetching job details: {str(e)}")
            return details

        finally:
            if owns_driver:
                driver.quit()

    def fetch_job_description(self, job_url, driver=None):
        """Backward-compatible wrapper around fetch_job_details."""
        details = self.fetch_job_details(job_url, driver=driver)
        return details.get("title", ""), details.get("company", ""), details.get("description", "")

    def fetch_job_descriptions(self, jobs):
        """Fetch enriched details for multiple jobs using one browser session."""
        if not jobs:
            return jobs

        driver = self._configure_driver()
        try:
            for i, job in enumerate(jobs):
                logger.info(
                    f"Fetching details for job {i + 1}/{len(jobs)}: {job.get('title', 'Unknown')}"
                )
                details = self.fetch_job_details(
                    job["link"],
                    driver=driver,
                    fallback_title=job.get("title", ""),
                    fallback_location=job.get("location", ""),
                )
                for key, value in details.items():
                    if key == "description":
                        if value:
                            jobs[i][key] = value
                    elif value not in ("", None):
                        jobs[i][key] = value
                time.sleep(1)
        finally:
            driver.quit()

        return jobs
    
    def save_jobs_to_excel(self, jobs, filename=None):
        """Save scraped jobs to an Excel file."""
        from job_apply_ai.scraper.jobs_io import save_jobs_to_excel
        return save_jobs_to_excel(jobs, filename)


def main():
    """
    Main function to demonstrate the LinkedIn scraper.
    """
    keyword = input("Enter job title (e.g., Software Engineer): ")
    location = input("Enter location (e.g., Remote, New York, Berlin): ")
    
    scraper = LinkedInScraper(headless=True)
    jobs = scraper.scrape_job_listings(keyword, location)
    
    if jobs:
        scraper.fetch_job_descriptions(jobs)
        
        # Save to Excel
        filename = scraper.save_jobs_to_excel(jobs)
        print(f"\n✅ Jobs saved to {filename}")
    else:
        print("\n❌ No LinkedIn jobs found.")


if __name__ == "__main__":
    main() 