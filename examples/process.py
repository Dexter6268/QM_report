import re
from collections import OrderedDict


def extract_and_reorganize_references(markdown_text):
    """
    处理Markdown文档中的引用资料，实现以下功能：
    1. 按照正文中引用的顺序排序引用资料
    2. 将正文中的引用替换为新的引用格式，按顺序从[1]开始编号
    3. 将内容相同的引用资料合并，更新正文中的引用编号
    4. 将所有引用资料放在文档末尾的"资料来源"章节中

    Args:
        markdown_text (str): 输入的Markdown文本

    Returns:
        str: 处理后的Markdown文本
    """
    # 提取所有资料来源章节
    source_sections = re.findall(r"### 资料来源\n((?:[^\n#]|(?:\n(?!#{1,3})))+)", markdown_text)

    # 提取所有资料来源条目及其对应的章节信息
    all_references = []
    section_pattern = r"## 第(\d+)节.*?\n"
    chapter_pattern = r"# 第(\d+)章.*?\n"

    for section in source_sections:
        # 找到此资料来源章节所属的节和章
        section_text_before = markdown_text[: markdown_text.find(section)]
        section_matches = list(re.finditer(section_pattern, section_text_before))
        chapter_matches = list(re.finditer(chapter_pattern, section_text_before))

        if section_matches:
            section_num = section_matches[-1].group(1)
        else:
            section_num = "0"

        if chapter_matches:
            chapter_num = chapter_matches[-1].group(1)
        else:
            chapter_num = "0"

        # 提取引用条目
        references = re.findall(r"\[(.*?)\](.*?)(?=\n\[|\n*$)", section, re.DOTALL)

        for ref_id, ref_content in references:
            all_references.append(
                {
                    "chapter": chapter_num,
                    "section": section_num,
                    "old_id": ref_id,
                    "content": ref_content.strip(),
                }
            )

    # 创建内容到引用的映射，用于合并相同内容的引用
    content_to_ref = {}
    for ref in all_references:
        content = ref["content"]
        if content not in content_to_ref:
            content_to_ref[content] = ref

    # 创建旧引用ID到新引用ID的映射
    old_to_new_id = {}

    # 提取正文中的所有引用，保持原始顺序
    citation_pattern = r"\[(\d+)\]"
    citations = re.findall(citation_pattern, markdown_text)

    # 按正文中引用的顺序创建有序字典
    ordered_refs = OrderedDict()
    new_id = 1

    for old_id in citations:
        for ref in all_references:
            if ref["old_id"] == old_id:
                content = ref["content"]
                if content not in ordered_refs.values():
                    # 如果内容是第一次出现，分配新的引用ID
                    old_to_new_id[old_id] = str(new_id)
                    ordered_refs[str(new_id)] = content
                    new_id += 1
                else:
                    # 如果内容已存在，找到对应的新ID
                    for new_ref_id, new_ref_content in ordered_refs.items():
                        if new_ref_content == content:
                            old_to_new_id[old_id] = new_ref_id
                            break
                break

    # 替换正文中的引用
    new_text = markdown_text
    for old_id, new_id in old_to_new_id.items():
        new_text = re.sub(r"\[" + re.escape(old_id) + r"\]", f"[{new_id}]", new_text)

    # 生成新的资料来源部分
    references_section = "# 资料来源\n"
    for ref_id, ref_content in ordered_refs.items():
        references_section += f"[{ref_id}] {ref_content}\n"

    # 从原文中删除所有的资料来源章节
    new_text = re.sub(r"### 资料来源\n((?:[^\n#]|(?:\n(?!#{1,3})))+)", "", new_text)

    # 添加新的资料来源章节到文档末尾
    new_text = new_text.rstrip() + "\n\n" + references_section

    return new_text


# 读取输入文件
with open("examples/test.md", "r", encoding="utf-8") as f:
    content = f.read()

# 处理参考文献
processed_content = extract_and_reorganize_references(content)

# 写入输出文件
with open("examples/test_processed.md", "w", encoding="utf-8") as f:
    f.write(processed_content)

print("参考文献处理完成！结果已保存到examples/test_processed.md")
