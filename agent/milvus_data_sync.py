# 将各种菜品数据同步到向量数据库中，这样可以从语义层面进行相似度的检索
import pymysql
from pymilvus import MilvusClient, DataType
import os
import torch
from pymysql.cursors import DictCursor
import sys
from pathlib import Path
ROOT_PATH= Path(__file__).parent.parent # 获取项目根目录
sys.path.insert(0, str(ROOT_PATH)) # 将项目的根目录添加到系统环境变量中

# 如何引入其他模型的代码,先需要确保这个模块已经添加到项目目录中
from utils.milvus_utils import get_milvus_client, get_embedding
from dotenv import  load_dotenv

load_dotenv()

# 1. 菜品数据同步到向量数据库中
def insert_data():
    """
    向向量数据库中插入数据的函数
    1. 创建向量数据库集合
    2. 生成文本向量并插入数据
    """
    # 1. 创建向量数据库集合
    client = get_milvus_client()
    collection_name = os.getenv("MILVUS_COLLECTION")  # 定义集合名称
    # 如果集合已存在，则删除它
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    #   1.2 确定集合中的字段及类型
    schema = MilvusClient.create_schema(auto_id=True)  # 创建自动生成ID的schema
    # 添加主键字段id
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    # 添加向量字段，维度为768
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
    # 添加文本字段，最大长度为256
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
    #   1.3 定义索引类型
    index_params = MilvusClient.prepare_index_params()
    # 为向量字段添加HNSW索引，使用IP（内积）作为度量方式
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="IP"  # cosine / ip（如果embedding模型以及完成了归一化，则建议选择IP效率高）
    )

    #   1.4 创建集合
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    # 2. 添加数据
    main_dishes = []
    # 准备要插入的文本数据:从业务数据库中查询
    sql = """
        select
            concat(
                '菜品名称:',ifnull(menu_items.dish_name, ''),';',
                '价格:', ifnull(menu_items.price, ''),';',
                '菜品类别:', ifnull(menu_items.category, ''),';',
                '描述:', ifnull(menu_items.description, ''),';',
                '辣度:', 
                case when menu_items.spice_level = 0 then '不辣'
                     when menu_items.spice_level = 1 then '微辣'
                     when menu_items.spice_level = 2 then '中辣'
                     when menu_items.spice_level = 3 then '重辣'
                     else '无辣'
                 end, ';',
                '口味:', ifnull(menu_items.flavor, ''),';',
                '主料:', ifnull(menu_items.main_ingredients, ''),';',
                '烹饪方法:', ifnull(menu_items.cooking_method, ''),';',
                '是否素食:', ifnull(menu_items.is_vegetarian, ''),';',
                '过敏源:', ifnull(menu_items.allergens, '')
            ) as text
        from menu_items
        """
    with pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"), ) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()#  ["菜品名称:宫保鸡丁;价格:28", "菜品名称:鱼香肉丝,价格:28"]
            main_dishes = [row["text"] for row in result]
    # print(main_dishes) # 打印菜品


    # 2.1 生成向量
    # 初始化HuggingFace嵌入模型
    embedding = get_embedding()

    # 将文本转换为向量
    vector_list = embedding.embed_documents(main_dishes)
    # 准备要插入的数据，包含向量和文本
    data_to_insert = [
        {"vector": vec, "text": txt} for vec, txt in zip(vector_list, main_dishes)
    ]

    # 将数据插入集合
    all_ids = client.insert(
        collection_name=collection_name,
        data=data_to_insert
    )
    print(all_ids)  # 打印插入数据的ID列表

if __name__ == '__main__':
    insert_data()