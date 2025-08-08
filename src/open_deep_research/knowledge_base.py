import os
import json
import time
import asyncio
import logging
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict


from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore


class KnowledgeBaseManager:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.similarity_threshold = 0.75  # 相似度阈值
        self.project_root = Path(__file__).parent.parent.parent
        self.kb_dir = self.project_root / "knowledge_base"
        # self.kb_dir.mkdir(exist_ok=True, parents=True)
        self.index_file = self.kb_dir / "kb_index.json"

    async def _load_kb_index(self) -> Dict[str, Any]:
        """加载知识库索引"""
        if self.index_file.exists():

            def read_json():
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            return await asyncio.to_thread(read_json)
        return {"knowledge_bases": []}

    async def _save_kb_index(self, index_data: Dict[str, Any]) -> None:
        """保存知识库索引"""

        def write_json():
            # 生成唯一的临时文件名
            timestamp = int(time.time() * 1000000)
            process_id = os.getpid()
            temp_file = self.index_file.with_suffix(f".tmp.{process_id}.{timestamp}")

            try:
                # 写入临时文件
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(index_data, f, ensure_ascii=False, indent=2)

                # 原子性替换
                temp_file.replace(self.index_file)

            except Exception as e:
                # 清理临时文件
                if temp_file.exists():
                    temp_file.unlink()
                raise e

        await asyncio.to_thread(write_json)

    def _compute_query_embedding(self, query: str) -> List[float]:
        """计算查询的向量表示"""
        return self.embeddings.embed_query(query)

    async def find_best_knowledge_base(self, query: str) -> Optional[tuple[str, float]]:
        """找到最匹配的知识库"""
        index_data = await self._load_kb_index()

        if not index_data["knowledge_bases"]:
            return None

        query_embedding = np.array([self._compute_query_embedding(query)])

        best_kb_id = ""
        best_similarity = 0.0

        for kb_info in index_data["knowledge_bases"]:
            # 计算与知识库代表性查询的相似度
            kb_embedding = np.array([kb_info["query_embedding"]])
            similarity: float = cosine_similarity(query_embedding, kb_embedding)[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_kb_id: str = kb_info["kb_id"]

        if best_similarity >= self.similarity_threshold:
            return best_kb_id, best_similarity

        return None

    async def create_knowledge_base(self, query: str) -> str:
        """创建新的知识库"""
        kb_id = hashlib.md5(f"{query}_{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:12]
        query_embedding = self._compute_query_embedding(query)

        index_data = await self._load_kb_index()
        index_data["knowledge_bases"].append(
            {
                "kb_id": kb_id,
                "representative_query": query,
                "query_embedding": query_embedding,
                "created_at": datetime.datetime.now().isoformat(),
                "query_count": 1,
                "doc_count": 0,
                "related_queries": [query],
            }
        )

        await self._save_kb_index(index_data)
        return kb_id

    async def update_knowledge_base_info(self, kb_id: str, query: str, doc_count: int):
        """更新知识库信息"""
        index_data = await self._load_kb_index()

        for kb_info in index_data["knowledge_bases"]:
            if kb_info["kb_id"] == kb_id:
                kb_info["query_count"] += 1
                kb_info["doc_count"] = doc_count
                kb_info["related_queries"].append(query)
                kb_info["last_updated"] = datetime.datetime.now().isoformat()

                # 如果查询数量超过阈值，重新计算代表性向量
                if kb_info["query_count"] % 5 == 0:
                    all_queries = kb_info["related_queries"]
                    all_embeddings = [self._compute_query_embedding(q) for q in all_queries]
                    # 使用平均向量作为代表性向量
                    kb_info["query_embedding"] = np.mean(all_embeddings, axis=0).tolist()

                break

        await self._save_kb_index(index_data)

    def get_knowledge_base_path(self, kb_id: str) -> Path:
        """获取知识库文件路径"""
        return self.kb_dir / f"kb_{kb_id}.pkl"

    async def save_knowledge_base(self, kb_id: str, vector_store: VectorStore, query: str, documents: List[Document]):
        """保存知识库"""

        def write_pickle():
            timestamp = int(time.time() * 1000000)
            process_id = os.getpid()
            kb_path = self.get_knowledge_base_path(kb_id)
            temp_path = kb_path.with_suffix(f".tmp.{process_id}.{timestamp}")

            save_data = {
                "vector_store": vector_store,
                "documents": documents,
                "kb_id": kb_id,
                "last_query": query,
                "updated_at": datetime.datetime.now().isoformat(),
                "doc_count": len(documents),
            }

            try:
                with open(temp_path, "wb") as f:
                    pickle.dump(save_data, f)

                # 原子性替换
                temp_path.replace(kb_path)

            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise e

        try:
            await asyncio.to_thread(write_pickle)
            await self.update_knowledge_base_info(kb_id, query, len(documents))
            logging.info(f"Knowledge base {kb_id} saved with {len(documents)} documents")
        except Exception as e:
            logging.error(f"Failed to save knowledge base {kb_id}: {e}")

    async def load_or_create_knowledge_base(
        self, query: str
    ) -> tuple[Optional[VectorStore], List[Document], str, bool]:
        """
        加载或创建知识库

        Returns:
            Tuple[vector_store, documents, kb_id, is_new]
        """
        # 尝试找到最匹配的知识库
        match_result = await self.find_best_knowledge_base(query)

        if match_result:
            kb_id, similarity = match_result
            logging.info(f"Found matching knowledge base {kb_id} with similarity {similarity:.3f}")

            # 加载现有知识库
            kb_path = self.get_knowledge_base_path(kb_id)
            if kb_path.exists():
                try:
                    save_data = await asyncio.to_thread(lambda: pickle.load(open(kb_path, "rb")))

                    vector_store = save_data["vector_store"]
                    documents = save_data["documents"]

                    logging.info(f"Loaded existing knowledge base {kb_id} with {len(documents)} documents")
                    return (
                        vector_store,
                        documents,
                        kb_id,
                        False,
                    )  # is_new = False，表示使用现有知识库
                except Exception as e:
                    logging.error(f"Failed to load knowledge base {kb_id}: {e}")

        # 创建新知识库
        kb_id = await self.create_knowledge_base(query)
        logging.info(f"No Matching results, created new knowledge base {kb_id} for query: {query}")
        return None, [], kb_id, True  # is_new = True，表示新建知识库

    async def print_all_knowledge_bases(
        self, show_full_content: bool = False, max_content_length: int = 200, max_url_num: int = 3
    ) -> None:
        """
        打印所有知识库的详细信息

        Args:
            show_full_content (bool): 是否显示完整的文档内容
            max_content_length (int): 当show_full_content=False时，每个文档显示的最大字符数
        """
        index_data = await self._load_kb_index()

        if not index_data["knowledge_bases"]:
            print("❌ 没有找到任何知识库")
            return

        print("=" * 100)
        print(f"📚 知识库总览 ({len(index_data['knowledge_bases'])} 个知识库)")
        print("=" * 100)

        for i, kb_info in enumerate(index_data["knowledge_bases"], 1):
            kb_id = kb_info["kb_id"]

            print(f"\n🏛️  知识库 {i}: {kb_id}")
            print("-" * 80)
            print(f"📝 代表性查询: {kb_info['representative_query']}")
            print(f"📅 创建时间: {kb_info['created_at']}")
            print(f"🔄 查询次数: {kb_info['query_count']}")
            print(f"📄 文档数量: {kb_info['doc_count']}")
            print(f"🕒 最后更新: {kb_info.get('last_updated', '未更新')}")

            # 显示相关查询
            if len(kb_info["related_queries"]) > 1:
                print(f"🔗 相关查询 ({len(kb_info['related_queries'])}):")
                for j, query in enumerate(kb_info["related_queries"][:5], 1):  # 最多显示5个
                    print(f"  {j}. {query}")
                if len(kb_info["related_queries"]) > 5:
                    print(f"  ... 还有 {len(kb_info['related_queries']) - 5} 个查询")

            # 尝试加载并显示文档内容
            kb_path = self.get_knowledge_base_path(kb_id)
            if kb_path.exists():
                try:
                    with open(kb_path, "rb") as f:
                        save_data = pickle.load(f)

                    documents = save_data.get("documents", [])
                    if documents:
                        print(f"\n📚 文档内容 ({len(documents)} 个文档):")

                        # 按URL分组显示文档
                        url_groups = defaultdict(list)
                        for doc in documents:
                            url = doc.metadata.get("url", "未知URL")
                            url_groups[url].append(doc)

                        for i, (url, docs) in enumerate(list(url_groups.items())[:max_url_num], 1):
                            print(f"\n📚 文档 {i}\n" + "-" * 80)
                            print(f"\n🌐 来源: {url}")
                            print(f"📑 标题: {docs[0].metadata.get('title', '未知标题')}")
                            print(f"📊 文档片段数: {len(docs)}")

                            # 显示第一个文档片段的内容
                            if docs:
                                content = docs[0].page_content
                                if show_full_content:
                                    print(f"📄 内容:\n```\n{content}\n```")
                                else:
                                    if len(content) > max_content_length:
                                        truncated_content = content[:max_content_length] + "..."
                                        print(f"📄 内容预览:\n```\n{truncated_content}\n```")
                                    else:
                                        print(f"📄 内容:\n```\n{content}\n```")

                        if len(url_groups) > max_url_num:
                            print(f"\n💡 还有 {len(url_groups) - max_url_num} 个其他来源的文档...")
                    else:
                        print("\n❌ 知识库中没有文档内容")

                except Exception as e:
                    print(f"\n❌ 无法加载知识库文件: {e}")
            else:
                print(f"\n❌ 知识库文件不存在: {kb_path}")

            print("\n" + "=" * 100)

    async def print_knowledge_base_summary(self) -> None:
        """打印知识库的简要统计信息"""
        index_data = await self._load_kb_index()

        if not index_data["knowledge_bases"]:
            print("❌ 没有找到任何知识库")
            return

        total_kbs = len(index_data["knowledge_bases"])
        total_docs = sum(kb["doc_count"] for kb in index_data["knowledge_bases"])
        total_queries = sum(kb["query_count"] for kb in index_data["knowledge_bases"])

        print("📊 知识库统计信息")
        print("=" * 50)
        print(f"🏛️  总知识库数量: {total_kbs}")
        print(f"📄 总文档数量: {total_docs}")
        print(f"🔍 总查询次数: {total_queries}")
        print(f"📈 平均每个知识库文档数: {total_docs / total_kbs:.1f}")
        print(f"🔄 平均每个知识库查询次数: {total_queries / total_kbs:.1f}")

        # 显示最活跃的知识库
        most_used_kb = max(index_data["knowledge_bases"], key=lambda x: x["query_count"])
        largest_kb = max(index_data["knowledge_bases"], key=lambda x: x["doc_count"])

        print(f"\n🏆 最活跃知识库:")
        print(f"   查询: {most_used_kb['representative_query']}")
        print(f"   使用次数: {most_used_kb['query_count']}")

        print(f"\n📚 最大知识库:")
        print(f"   查询: {largest_kb['representative_query']}")
        print(f"   文档数量: {largest_kb['doc_count']}")

    async def inspect_knowledge_base(
        self, kb_id: str, show_full_content: bool = True, max_content_length: int = 500
    ) -> None:
        """
        查看特定知识库的详细信息

        Args:
            kb_id (str): 知识库ID
            show_full_content (bool): 是否显示完整的文档内容
            max_content_length (int): 当show_full_content=False时，每个文档显示的最大字符数
        """
        # 先从索引中查找知识库信息
        index_data = await self._load_kb_index()
        kb_info = None

        for kb in index_data["knowledge_bases"]:
            if kb["kb_id"] == kb_id:
                kb_info = kb
                break

        if not kb_info:
            print(f"❌ 未找到知识库 ID: {kb_id}")
            return

        kb_path = self.get_knowledge_base_path(kb_id)

        if not kb_path.exists():
            print(f"❌ 知识库文件不存在: {kb_path}")
            return

        try:
            with open(kb_path, "rb") as f:
                save_data = pickle.load(f)

            print("=" * 80)
            print(f"📚 知识库详细信息: {kb_id}")
            print("=" * 80)
            print(f"📝 代表性查询: {kb_info['representative_query']}")
            print(f"📅 创建时间: {kb_info['created_at']}")
            print(f"🔄 查询次数: {kb_info['query_count']}")
            print(f"📄 文档数量: {save_data.get('doc_count', len(save_data.get('documents', [])))}")
            print(f"🕒 最后更新: {save_data.get('updated_at', '未知')}")
            print(f"📁 文件路径: {kb_path}")
            print(f"💾 文件大小: {kb_path.stat().st_size / 1024:.2f} KB")

            # 显示所有相关查询
            print(f"\n🔗 所有相关查询 ({len(kb_info['related_queries'])}):")
            for i, query in enumerate(kb_info["related_queries"], 1):
                print(f"  {i:2d}. {query}")

            documents = save_data.get("documents", [])
            if not documents:
                print("\n❌ 知识库中没有文档")
                return

            print(f"\n📄 文档详情 ({len(documents)} 个文档):")
            print("=" * 80)

            for i, doc in enumerate(documents, 1):
                print(f"\n📑 文档 {i}:")
                print("-" * 60)

                # 显示元数据
                if hasattr(doc, "metadata") and doc.metadata:
                    print("📋 元数据:")
                    for key, value in doc.metadata.items():
                        print(f"  {key}: {value}")

                # 显示文档内容
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                    print(f"\n📝 内容 (长度: {len(content)} 字符):")

                    if show_full_content:
                        print(f"```\n{content}\n```")
                    else:
                        # 显示截断的内容
                        if len(content) > max_content_length:
                            truncated_content = content[:max_content_length] + "..."
                            print(f"```\n{truncated_content}\n```")
                            print(f"💡 内容已截断，完整内容共 {len(content)} 字符")
                        else:
                            print(f"```\n{content}\n```")
                else:
                    print("❌ 文档没有内容")

            print("=" * 80)

        except Exception as e:
            print(f"❌ 读取知识库文件失败: {e}")

    async def list_knowledge_base_ids(self) -> List[str]:
        """返回所有知识库的ID列表"""
        index_data = await self._load_kb_index()
        return [kb["kb_id"] for kb in index_data["knowledge_bases"]]


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
    kbm = KnowledgeBaseManager(embeddings)
    asyncio.run(kbm.print_all_knowledge_bases(show_full_content=True, max_url_num=50))
