"""Shared browser configuration for Selenium-based job scrapers."""

import os

import undetected_chromedriver as uc
from selenium import webdriver


def create_chrome_driver(headless=True):
    """Configure and return a Chrome WebDriver."""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    browser_paths = [
        os.environ.get("CHROME_BINARY"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Google\Chrome Dev\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    browser_executable_path = next(
        (path for path in browser_paths if path and os.path.isfile(path)),
        None,
    )

    driver_kwargs = {"options": options, "use_subprocess": True}
    if browser_executable_path:
        driver_kwargs["browser_executable_path"] = browser_executable_path

    cached_driver = os.path.expanduser(
        r"~\appdata\roaming\undetected_chromedriver\undetected_chromedriver.exe"
    )
    if os.path.isfile(cached_driver):
        driver_kwargs["driver_executable_path"] = cached_driver

    return uc.Chrome(**driver_kwargs)
