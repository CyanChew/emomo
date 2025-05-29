import os
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class MolmoWrapper:
    def __init__(self, chromedriver_path="/usr/local/bin/chromedriver", headless=False, max_retries=3):
        self.chromedriver_path = chromedriver_path
        self.headless = headless
        self.max_retries = max_retries
        self.driver = None

    def _human_delay(self, a=0.5, b=1.5):
        time.sleep(random.uniform(a, b))

    def _setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        service = Service(executable_path=self.chromedriver_path)
        return webdriver.Chrome(service=service, options=chrome_options)

    def _disable_tracking(self):
        self.driver.execute_script("""
            if (window.heap && typeof window.heap.track === 'function') {
                window.heap.track = function() {};
            } else {
                window.heap = {};
                window.heap.track = function() {};
            }
        """)
        self.driver.execute_script("""Object.defineProperty(navigator, 'webdriver', {get: () => undefined});""")

    def point_to_object(self, image_path, prompt="object"):
        for attempt in range(self.max_retries):
            print(f"\nAttempt {attempt + 1}/{self.max_retries}")
            self.driver = self._setup_driver()
            self.driver.get("https://playground.allenai.org/?model=mm-olmo-uber-model-v4-synthetic")
            self._disable_tracking()
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
                )
                upload_button = self.driver.find_element(By.XPATH, "//input[@type='file']")
                upload_button.send_keys(os.path.abspath(image_path))
                self._human_delay()

                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Message Molmo']"))
                )
                prompt_box = self.driver.find_element(By.XPATH, "//textarea[@placeholder='Message Molmo']")
                prompt_box.clear()
                prompt_box.send_keys(f"point to the {prompt}")
                prompt_box.send_keys(Keys.RETURN)
                self._human_delay()

                # Captcha check (safe JS)
                try:
                    captcha_error = driver.execute_script("""
                        return Array.from(document.querySelectorAll('*'))
                            .some(el => el.innerText.includes('captchaToken: Field required'));
                    """)
                except:
                    captcha_error = False

                if captcha_error:
                    print("Captcha detected, retrying...")
                    self.driver.quit()
                    continue

                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "svg circle"))
                )
                self._human_delay()

                coords = self.driver.execute_script("""
                    const circle = document.querySelector('svg circle');
                    return circle ? { cx: circle.getAttribute('cx'), cy: circle.getAttribute('cy') } : null;
                """)

                if coords:
                    print(f"Success! Circle coordinates: cx = {coords['cx']}, cy = {coords['cy']}")
                    return coords
                else:
                    print("Circle not found after success.")
                    return None

            except Exception as e:
                print(f"Error: {e}")
                self.driver.quit()
                time.sleep(2)

        print("Failed after maximum retries.")
        return None

    def kill_molmo(self):
        if self.driver is not None:
            self.driver.quit()
            self.driver = None



