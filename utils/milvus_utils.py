import torch
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient
import os


def get_milvus_client():
    client = MilvusClient(
        uri=os.getenv("MILVUS_URI"),  # 从环境变量获取Milvus服务器URI
        token=os.getenv("MILVUS_TOKEN"),  # 从环境变量获取认证令牌
    )
    return client


def get_embedding():
    embedding = HuggingFaceEmbeddings(
        model_name="./model/bge-base-zh",  # 指定本地模型路径
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu"  # 根据CUDA可用性选择设备
        },
        encode_kwargs={"normalize_embeddings": True}  # 模型进行归一化， 配合IP
    )
    return embedding