import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # for debugging
        page = await browser.new_page()
        await page.goto("https://www.public.nm.eurocontrol.int/PUBPORTAL/gateway/spec/")
        await page.wait_for_load_state("networkidle")

        # Retry logic: check if function exists
        for _ in range(20):  # retry for a few seconds
            function_exists = await page.evaluate("() => typeof Tl === 'function'")
            if function_exists:
                break
            await asyncio.sleep(0.5)
        else:
            raise Exception("Function 'Tl' not found on the page.")

        # Now run it with an appropriate argument if needed
        result = await page.evaluate("() => Tl('your-arg')()")  # Replace 'your-arg'
        print(result)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
