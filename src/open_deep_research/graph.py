import logging
from datetime import datetime
from typing import Any, Literal, cast, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command, Send

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    SearchQuery,
    Feedback,
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions,
    section_writer_instructions,
    final_planner_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs,
    compile_final_report_instructions,
)

from open_deep_research.configuration import WorkflowConfiguration
from open_deep_research.utils import (
    reduce_source_str,
    get_model,
    format_sections,
    get_config_value,
    get_search_params,
    select_and_execute_search,
    get_today_str,
    process_references_from_sections,
)

## Nodes --


async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections.

    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections

    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.

    Returns:
        Dict containing the generated sections
    """

    # Inputs
    topic = state["topic"]

    # Get list of feedback on the report plan
    feedback_list = state.get("feedback_on_report_plan", [])

    # Concatenate feedback on the report plan into a single string
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = (
        configurable.search_api_config or {}
    )  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing)
    writer_model = get_model(configurable, "writer")
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        today=get_today_str(),
    )

    # Generate queries
    results = cast(
        Queries,
        await structured_llm.ainvoke(
            [
                SystemMessage(content=system_instructions_query),
                HumanMessage(content="生成能够辅助制定报告章节结构的搜索关键词。"),
            ]
        ),
    )

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)
    source_str = await reduce_source_str(source_str, max_tokens=10000)

    # Set the planner
    planner_llm = get_model(configurable, "planner")
    structured_llm = planner_llm.with_structured_output(Sections)
    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(
        topic=topic, report_organization=report_structure, context=source_str, feedback=feedback
    )
    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, chapter_name, description, research, and content fields."""

    # Generate the report sections
    report_sections = cast(
        Sections,
        await structured_llm.ainvoke(
            [
                SystemMessage(content=system_instructions_sections),
                HumanMessage(content=planner_message),
            ]
        ),
    )

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}


def human_feedback(
    state: ReportState, config: RunnableConfig
) -> Command[Literal["generate_report_plan", "build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps.

    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided

    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow

    Returns:
        Command to either regenerate plan or start section writing
    """

    # Get sections
    topic = state["topic"]
    sections = state["sections"]
    chapter_no = set()
    i = 0
    k = 0
    sections_str = ""
    for j, section in enumerate(sections):
        if section.chapter_name not in chapter_no:
            chapter_no.add(section.chapter_name)
            i += 1
            k = j
            sections_str += (
                f"第{i}章 {section.chapter_name}\n"
                + f"\t第{j + 1 - k}节 {section.name}\n"
                + f"\t\t概要: {section.description}\n"
                + f"\t\t是否需要联网搜索: {'是' if section.research else '否'}\n"
            )
        else:
            sections_str += (
                f"\t第{j + 1 - k}节 {section.name}\n"
                + f"\t\t概要: {section.description}\n"
                + f"\t\t是否需要联网搜索: {'是' if section.research else '否'}\n"
            )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""请对以下报告框架提供反馈。\n\n{sections_str}\n
    \n该报告框架是否符合您的需求？\n输入'true'以确认采用该报告框架。\n或提供修改意见以重新生成报告框架："""

    feedback = interrupt(interrupt_message)
    if isinstance(feedback, dict):
        feedback = list(feedback.values())[0]
        logging.warning(
            f"feedback = {feedback}; type = {type(feedback)}"
        )  # Get the first value if feedback is a dict
    # If the user approves the report plan, kick off section writing
    if (isinstance(feedback, bool) and feedback is True) or (
        isinstance(feedback, str) and feedback.lower() == "true"
    ):
        # Treat this as approve and kick off section writing
        return Command(
            goto=[
                Send(
                    "build_section_with_web_research",
                    {"topic": topic, "section": s, "search_iterations": 0},
                )
                for s in sections
                if s.research
            ]
        )

    # If the user provides feedback, regenerate the report plan
    elif isinstance(feedback, str):
        # Treat this as feedback and append it to the existing list
        return Command(goto="generate_report_plan", update={"feedback_on_report_plan": [feedback]})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


async def generate_queries(
    state: SectionState, config: RunnableConfig
) -> Dict[str, List[SearchQuery]]:
    """Generate search queries for researching a specific section.

    This node uses an LLM to generate targeted search queries based on the
    section topic and description.

    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate

    Returns:
        Dict containing the generated search queries
    """

    # Get state
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries
    writer_model = get_model(configurable, "writer")
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(
        topic=topic,
        chapter_name=section.chapter_name,
        section_name=section.name,
        section_topic=section.description,
        number_of_queries=number_of_queries,
        today=get_today_str(),
    )

    # Generate queries
    queries = cast(
        Queries,
        await structured_llm.ainvoke(
            [
                SystemMessage(content=system_instructions),
                HumanMessage(content="针对给定的系统提示生成浏览器搜索的关键词或语句"),
            ]
        ),
    )

    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig) -> Dict[str, str | int]:
    """Execute web searches for the section queries.

    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context

    Args:
        state: Current state with search queries
        config: Search API configuration

    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = (
        configurable.search_api_config or {}
    )  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters
    max_tokens = configurable.max_tokens
    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(
        search_api, query_list, params_to_pass, max_tokens=max_tokens
    )
    source_str = await reduce_source_str(
        source_str, max_tokens=max_tokens
    )  # Reduce source string if too long
    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}


async def write_section(state: SectionState, config: RunnableConfig) -> Command:
    """Write a section of the report and evaluate if more research is needed.

    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails

    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation

    Returns:
        Command to either complete section or do more research
    """

    # Get state
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(
        topic=topic,
        chapter_name=section.chapter_name,
        section_name=section.name,
        section_topic=section.description,
        context=source_str,
        section_content=section.content,
    )

    # Generate section
    writer_model = get_model(configurable, "writer")
    section_content = await writer_model.ainvoke(
        [
            SystemMessage(content=section_writer_instructions),
            HumanMessage(content=section_writer_inputs_formatted),
        ]
    )

    # Write content to the section object
    section.content = cast(str, section_content.content)

    # Grade prompt
    section_grader_message = "评估报告质量并提出补充信息所需的后续问题。若评估结果为'通过'(pass)，则所有后续查询返回空字符串。若评估结果为'不通过'(fail)，需提供具体的搜索查询以获取缺失信息。"

    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic,
        chapter_name=section.chapter_name,
        section_name=section.name,
        section_topic=section.description,
        section=section.content,
        number_of_follow_up_queries=configurable.number_of_queries,
    )

    # Use planner model for reflection
    reflection_model = get_model(configurable, "planner").with_structured_output(Feedback)
    # Generate feedback
    feedback = cast(
        Feedback,
        await reflection_model.ainvoke(
            [
                SystemMessage(content=section_grader_instructions_formatted),
                HumanMessage(content=section_grader_message),
            ]
        ),
    )

    # If the section is passing or the max search depth is reached, publish the section to completed sections
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections
        update: Dict[str, Any] = {"completed_sections": [section]}
        if configurable.include_source_str:
            update["source_str"] = source_str
        return Command(update=update, goto=END)

    # Update the existing section with new content and update search queries
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web",
        )


async def generate_conclusion_plan(state: ReportState, config: RunnableConfig):
    """为最终报告的结论部分（最后两章）生成大纲。

    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model

    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    report_structure = configurable.report_structure
    conclusion_structure = configurable.conclusion_structure
    # Get state
    topic = state["topic"]
    sections = state["sections"]
    completed_report_sections = state["report_sections_from_research"]

    # set planner model
    planner_llm = get_model(configurable, "planner")
    structured_llm = planner_llm.with_structured_output(Sections)

    # Format system instructions
    system_instructions_sections = final_planner_instructions.format(
        topic=topic,
        report_organization=report_structure,
        conclusion_structure=conclusion_structure,
        context=completed_report_sections,
    )

    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections.
    Each section must have: name, chapter_name, description, research, and content fields."""
    # Generate the final plan
    report_sections = cast(
        Sections,
        await structured_llm.ainvoke(
            [
                SystemMessage(content=system_instructions_sections),
                HumanMessage(content=planner_message),
            ]
        ),
    )

    # Get sections
    sections += report_sections.sections

    # Write the updated section to completed sections
    return {"sections": sections}


async def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.

    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.

    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model

    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Get state
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]

    # Format system instructions
    system_instructions = final_section_writer_instructions.format(
        topic=topic,
        chapter_name=section.chapter_name,
        section_name=section.name,
        section_topic=section.description,
        context=completed_report_sections,
    )

    # Generate section
    writer_model = get_model(configurable, "writer")
    section_content = await writer_model.ainvoke(
        [
            SystemMessage(content=system_instructions),
            HumanMessage(content="根据给定的系统提示撰写报告节内容"),
        ]
    )

    # Write content to section
    section.content = cast(str, section_content.content)

    # Write the updated section to completed sections
    return {"completed_sections": [section]}


def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.

    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.

    Args:
        state: Current state with completed sections

    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections, final=True)

    return {"report_sections_from_research": completed_report_sections}


def compile_final_report(state: ReportState, config: RunnableConfig):
    """Compile all sections into the final report.

    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report

    Args:
        state: Current state with all completed sections

    Returns:
        Dict containing the complete report
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    references_section, sections = process_references_from_sections(sections)
    all_sections = format_sections(sections, final=True)
    final_report = all_sections + references_section

    timestamp = datetime.now().strftime("%Y-%m-%d-%H_%M")  # 格式：2025-07-18 14:30
    filename = f"examples/Report_at_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_report)

    if configurable.include_source_str:
        return {"final_report": final_report, "source_str": state["source_str"]}
    else:
        return {"final_report": final_report}


def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.

    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.

    Args:
        state: Current state with all sections and research context

    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send(
            "write_final_sections",
            {
                "topic": state["topic"],
                "section": s,
                "report_sections_from_research": state["report_sections_from_research"],
            },
        )
        for s in state["sections"]
        if not s.research
    ]


# Report section sub-graph --

# Add nodes
section_builder = StateGraph(SectionState, output_schema=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section --

# Add nodes
builder = StateGraph(
    ReportState,
    input_schema=ReportStateInput,
    output_schema=ReportStateOutput,
    config_schema=WorkflowConfiguration,
)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())  # type: ignore
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("generate_conclusion_plan", generate_conclusion_plan)
builder.add_node("write_final_sections", write_final_sections)  # type: ignore
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_edge("gather_completed_sections", "generate_conclusion_plan")
builder.add_conditional_edges(
    "generate_conclusion_plan", initiate_final_section_writing, ["write_final_sections"]  # type: ignore
)
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

print("builder compiling ...")
graph = builder.compile()
print("builder compiled.")
