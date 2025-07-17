# import requests
# url = "http://hnzd.stats.gov.cn/dcsj/sjfb/hns/zxfb/202503/t20250331_247703.html"

# res = requests.get(url)
# if res.status_code == 200:
#     print("Request was successful.")
#     res.encoding = res.apparent_encoding  # Ensure the response is decoded correctly
#     html = res.text
#     print(html)
#     print(res.apparent_encoding)
# else:
#     print(f"Request failed with status code: {res.status_code}")

import httpx
import asyncio

async def fetch_directly():
    async with httpx.AsyncClient() as client:
        res = await client.get(
            "http://hnzd.stats.gov.cn/dcsj/sjfb/hns/zxfb/202503/t20250331_247703.html"
        )
        # res.encoding = res.apparent_encoding  # 自动检测编码
        print(res.text)

asyncio.run(fetch_directly())