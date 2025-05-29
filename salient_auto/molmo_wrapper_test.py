import time
import os
import argparse
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def human_delay(a=0.5, b=1.5):
    time.sleep(random.uniform(a, b))

# ---- Argument parsing ----
parser = argparse.ArgumentParser(description="Run the Selenium script with optional headless mode.")
parser.add_argument('-i', '--image_path', default="teapot.jpg")
parser.add_argument('-p', '--prompt', default="teapot", help="Point to the ____")
parser.add_argument('--headless', action='store_true', help="Run the script in headless mode (not recommended since likely to have more captcha errors)")
args = parser.parse_args()

# ---- Setup ChromeDriver ----
chromedriver_path = "/usr/local/bin/chromedriver"
service = Service(executable_path=chromedriver_path)
chrome_options = Options()
if args.headless:
    chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")


# ---- Retry Loop ----
MAX_RETRIES = 3
success = False

for attempt in range(MAX_RETRIES):
    print(f"\nAttempt {attempt + 1}/{MAX_RETRIES}")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get("https://playground.allenai.org/?model=mm-olmo-uber-model-v4-synthetic")
    
    # Disable Heap
    driver.execute_script("""
        if (window.heap && typeof window.heap.track === 'function') {
            window.heap.track = function() {};
        } else {
            window.heap = {};
            window.heap.track = function() {};
        }
    """)
    # Hide automation
    driver.execute_script("""Object.defineProperty(navigator, 'webdriver', {get: () => undefined});""")

    try:
        # Upload image
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//input[@type='file']")))
        upload_button = driver.find_element(By.XPATH, "//input[@type='file']")
        upload_button.send_keys(os.path.abspath(args.image_path))
        human_delay()

        # Enter prompt
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Message Molmo']")))
        prompt_box = driver.find_element(By.XPATH, "//textarea[@placeholder='Message Molmo']")
        prompt_box.clear()
        prompt_box.send_keys(f"point to the {args.prompt}")
        prompt_box.send_keys(Keys.RETURN)
        human_delay()

        # Detect captcha error via JS
        try:
            captcha_error = driver.execute_script("""
                return Array.from(document.querySelectorAll('*'))
                    .some(el => el.innerText.includes('captchaToken: Field required'));
            """)
        except:
            captcha_error = False

        if captcha_error:
            print("Captcha detected, retrying...")
            continue  # go to next retry attempt

        # If no captcha, wait for result
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "svg circle"))
        )
        human_delay()

        # Get circle coordinates
        circle_coords = driver.execute_script("""
            var circle = document.querySelector('svg circle');
            return circle ? { cx: circle.getAttribute('cx'), cy: circle.getAttribute('cy') } : null;
        """)
        if circle_coords:
            print(f"Success! Circle coordinates: cx = {circle_coords['cx']}, cy = {circle_coords['cy']}")
        else:
            print("Circle not found after success.")
        success = True
        break  # Exit retry loop on success

    except Exception as e:
        time.sleep(2)
        print(f"Error during attempt {attempt + 1}: {e}")
        driver.quit()
        continue

if not success:
    print("\nFailed after maximum retries.")