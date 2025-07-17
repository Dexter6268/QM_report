import asyncio
import uuid 
import os
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(filename)s | %(funcName)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
load_dotenv("../.env")
print(os.environ["PLANNER_MODEL"])
print(os.environ["PLANNER_PROVIDER"])

REPORT_STRUCTURE = """ 
第一章、产品质量发展现状
   第一节、总体概况：结合当年的数据和政策分析整体情况
   第二节、地区概况：结合当年的数据和政策总体分析各地区的概况
   第三节、行业概况：结合当年的数据和政策总体分析各行业的概况
第二章、地区产品质量状况：包含若干节（对应于第一章第二节中列举的地区），每一节都聚焦于一个特定地区的产品质量状况，分析该地区的产品质量发展现状，若报告范围为全国则列举所有省份，若报告范围为某省份则列举所有地级市
第三章、行业产品质量状况：包含3-5节，每一节都聚焦于一个特定行业的产品质量状况，分析该行业的产品质量发展现状，比如装备制造类行业、资源加工类行业、食药烟酒类行业以及消费品工业制造业等等
第四章、问题分析：包括3-5节，结合前面章节的内容，分析当前产品质量发展中存在的问题和挑战
第五章、政策建议：包括3-5节，提出针对第四章中分析出的问题的解决方案和建议
"""
topic = "湖南省制造业产品质量合格率分析报告（2024年）"

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("PLANNER_MODEL", "deepseek-chat")
PROVIDER = os.environ.get("WRITER_PROVIDER", "deepseek")
BASE_URL = 'https://api.deepseek.com/v1'
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "tavily",
                           "search_api_config": {"api_key": TAVILY_API_KEY},
                           "planner_provider": PROVIDER,
                           "planner_model": DEEPSEEK_MODEL,
                           "planner_model_kwargs": {
                              "api_key": DEEPSEEK_API_KEY,
                              "base_url": BASE_URL
                           },
                           "writer_provider": PROVIDER,
                           "writer_model": DEEPSEEK_MODEL,
                           "writer_model_kwargs": {
                              "api_key": DEEPSEEK_API_KEY,
                              "base_url": BASE_URL
                           },
                           "summarization_model_provider": PROVIDER,
                           "summarization_model_model": DEEPSEEK_MODEL,
                           "summarization_model_kwargs": {
                              "api_key": DEEPSEEK_API_KEY,
                              "base_url": BASE_URL
                           },      
                           "max_search_depth": 2,
                           "report_structure": REPORT_STRUCTURE,
                           }}

# Define research topic about Model Context Protocol

async def main():
    # Run the graph workflow until first interruption (waiting for user feedback)
    async for event in graph.astream({"topic":topic,}, thread, stream_mode="debug"):
        if '__interrupt__' in event:
            interrupt_value = event['__interrupt__'][0].value
            #   display(Markdown(interrupt_value))
            print("Interrupt value:", interrupt_value)
    
    resume = input("接受还是修改，接受请输入True，修改请输入修改建议: ")
    if resume.lower() == "true":
        resume = True
    # async for event in graph.astream(Command(resume="第二章的地区不够多，把湖南省的地级市都包括，每一个地级市一节"), thread, stream_mode="updates"):
    #     if '__interrupt__' in event:
    #         interrupt_value = event['__interrupt__'][0].value
    #         print("Interrupt value:", interrupt_value)
    #         # display(Markdown(interrupt_value))
    i = 0
    async for event in graph.astream(Command(resume=resume), thread, stream_mode="updates"):
        logging.info(f"Event{i} finished}")
        i += 1
    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')
    # 将 report 字符串中的 \n 识别为换行，并输出为 .md 文件
    try:
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("报告已保存为 report.md")
    except Exception as e:
        print(f"保存报告时出错: {e}")

if __name__ == "__main__":
    
    asyncio.run(main())