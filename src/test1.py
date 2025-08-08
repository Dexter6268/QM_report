import os
import asyncio
from threading import local
from sentence_transformers import SentenceTransformer
from langchain.embeddings import init_embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# # 设置缓存目录（可选）
# os.environ["HF_HOME"] = "./models_cache"  # 自定义缓存目录
# # 或者使用默认缓存目录: ~/.cache/huggingface

# # 下载模型到本地
# model = SentenceTransformer("BAAI/bge-small-zh")

# print(f"模型已下载到: {model._modules['0'].auto_model.config.name_or_path}")
# print("模型文件位置:", os.path.expanduser("~/.cache/huggingface/transformers/"))

# embedding = init_embeddings("huggingface:BAAI/bge-small-zh", model_kwargs={"local_files_only": True})
# embedding = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh",
#     model_kwargs={"local_files_only": True}
# )


# 你的写法
async def test_your_approach():
    try:
        embeddings = await asyncio.to_thread(
            init_embeddings, "huggingface:BAAI/bge-small-zh", model_kwargs={"local_files_only": True}
        )
        print("✅ 你的写法是正确的")
        return embeddings
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(test_your_approach())
