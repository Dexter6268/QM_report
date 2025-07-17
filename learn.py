import asyncio
import time
from tqdm.asyncio import tqdm
from tavily import AsyncTavilyClient
from typing import List, Optional, Dict, Any, Union, Literal, Annotated, cast

# async def print_trash(i):
#     # print(f"Processing trash {i}...")
#     async for i in tqdm(range(50), desc=f"Task {i}"):
#         await asyncio.sleep(0.1)  # 模拟耗时操作

# async def coroutine_example():
#     tasks = [print_trash(i) for i in range(3)]
#     start = time.time()
#     await asyncio.gather(*tasks)
#     end = time.time()
#     print(f"All tasks completed in {end - start:.2f} seconds.")

# asyncio.run(coroutine_example())

async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True):
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

search_docs = asyncio.run(tavily_search_async(["Donald Trump", "Joe Biden", "Barack Obama"], max_results=5, topic="news", include_raw_content=True))
for i, doc in enumerate(search_docs):
    print(f'doc {i + 1}:')
    for j, result in enumerate(doc['results']):
        print(f"Result {j + 1}:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content: {result['content'][:100]}...")