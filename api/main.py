# 将后段能力抽象成具体接口供前端调用
import os
from typing import Optional, List, Literal

import pymysql
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymysql.cursors import DictCursor
from starlette.responses import StreamingResponse
from datetime import datetime
from difflib import SequenceMatcher
import re

from agent.langchain_assistant import assistant_query, get_delivery_info
from utils.redis_utils import get_redis_client

app = FastAPI()
# 定义正则表达式，用于移除文本中的标点符号
_PUNCT_RE = re.compile(r"[\s,，。.!！？?;；:：'\"“”‘’（）()\[\]【】<>《》、/\\|_\-]+")


# 对文本进行清洗的函数：
def _normalize_text(text: str) -> str:
    return _PUNCT_RE.sub("", (text or "").strip())


def get_similarity_score(query: str, faq_question: str):
    """判断用户的问题和FAQ的问题相似度得分"""
    query = _normalize_text(query)
    faq_question = _normalize_text(faq_question)
    # 1. 计算两个字符串的相似度得分
    similarity_score = SequenceMatcher(None, query, faq_question).ratio()
    query = set(list(query))
    faq_question = set(list(faq_question))
    # 2. 计算两个字符串的Jaccard相似度得分
    jaccard_score = len(query.intersection(faq_question)) / len(query.union(faq_question))

    # 3. 计算两个字符串的包含关系得分
    bonus = 0
    if query in faq_question or faq_question in query:
        bonus = 0.5
    return similarity_score * 0.6 + jaccard_score * 0.4 + bonus  # 根据实际需求调整权重


class FAQItem(BaseModel):
    id: str
    question: str
    answer: str


class ChatRequest(BaseModel):
    query: str = Field(description="用户输入的聊天内容")


class ReservationItem(BaseModel):
    id: int
    num_people: int
    num_children: int
    arrival_time: Optional[str] = None
    seat_preference: Optional[str] = None
    main_dish_preference: Optional[str] = None
    other_comments: Optional[str] = None
    created_at: Optional[str] = None


class ReservationListResponse(BaseModel):
    success: bool
    reservations: List[ReservationItem]
    count: int
    message: str


class FAQResponse(BaseModel):
    success: bool
    query: str
    suggestions: List[FAQItem]


class MenuItem(BaseModel):
    id: int
    dish_name: str
    price: float
    formatted_price: str  # 格式化后的价格，例如：¥12.50
    description: str
    category: str
    spice_level: int
    spice_text: str
    flavor: str
    main_ingredients: str
    cooking_method: str
    is_vegetarian: bool
    allergens: str
    is_available: bool


class MenuListResponse(BaseModel):
    success: bool
    menu_items: List[MenuItem]
    message: str
    count: int


class DeliveryRequest(BaseModel):
    address: str = Field(description="目标配送地址")
    travel_mode: str = Field(default="2", description="配送方式 1-骑行  2-驾车  3-直线距离")


class DeliveryResponse(BaseModel):
    success: bool = Field(description="是否成功")
    address: str = Field(description="配送地址")
    distance_km: float = Field(description="配送距离(单位：公里)")
    duration_min: int = Field(description="配送时间(单位：分钟)")
    in_range: bool = Field(description="是否在配送范围内")
    travel_mode: Literal["骑行", "驾车", "直线"] = Field(default="驾车", description="配送方式")
    message: str = Field(description="响应消息")


def load_faq_from_redis() -> list[FAQItem]:
    """从redis中读取所有的FAQ"""
    client = get_redis_client()
    faq_all_ids = client.smembers("faq:ids")
    full_keys = [f"faq:item:{id}" for id in faq_all_ids]
    pipeline = client.pipeline()
    for key in full_keys:
        pipeline.hgetall(key)
    faq_items = pipeline.execute()
    return [FAQItem(**item) for item in faq_items if item]


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """提供一个聊天接口"""
    # 以流式的方式返回给前端
    session_id = "session_001"
    return StreamingResponse(assistant_query(query=request.query, session_id=session_id),
                             media_type="text/event-stream")


# 提供预定接口
@app.get("/reservation/list", response_model=ReservationListResponse)
def book_endpoint(limit: int = 1):
    """获取所有预定的信息
        :arg
        limit: int, 获取预定信息的条数，咱这先不使用
    """
    sql = """
          select id, \
                 num_people, \
                 num_children, \
                 arrival_time, \
                 seat_preference, \
                 main_dish_preference, \
                 other_comments, \
                 created_at
          from reservation_order \
          """
    with pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"), ) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            item_list = []
            for item in result:
                item_list.append(ReservationItem(
                    id=item['id'],
                    num_people=item['num_people'],
                    num_children=item['num_children'],
                    arrival_time=datetime.strftime(item['arrival_time'], "%Y-%m-%d %H:%M:%S"),
                    seat_preference=item['seat_preference'],
                    main_dish_preference=item['main_dish_preference'],
                    other_comments=item['other_comments'],
                    created_at=datetime.strftime(item['created_at'], "%Y-%m-%d %H:%M:%S")
                ))
            return ReservationListResponse(success=True, reservations=item_list, count=len(item_list),
                                           message="success")


@app.get("/faq/suggest", response_model=FAQResponse)
def faq_suggest_endpoint(query: str, limit: int):
    """根据用户的问题从FAQ中进行匹配"""
    # 1. 从redis中读取所有的FAQ
    faq_items = load_faq_from_redis()
    """
    [
        FAQItem(id: str,question: str, answer: str),
        FAQItem(id: str,question: str, answer: str),
        FAQItem(id: str,question: str, answer: str)
    ]

    """
    # 2. 根据用户的问题进行匹配
    scores_list = []
    for faq_item in faq_items:
        score = get_similarity_score(query, faq_item.question)  # 将用户的问题和FAQ的问题进行匹配，计算得分
        scores_list.append((score, faq_item))  # 将得分和FAQ项放入列表中

    # 3. 根据得分进行排序，取出得分最高的前limit个FAQ项
    scores_list.sort(key=lambda x: x[0], reverse=True)
    limit = 2  # 根据实际需求调整
    top_k_faq_items = [faq_item for _, faq_item in scores_list[:limit]]

    # 4. 返回匹配的结果
    return FAQResponse(success=True, query=query, suggestions=top_k_faq_items)


@app.get("/menu/list", response_model=MenuListResponse)
def menu_list():
    sql = """
          select id, \
                 dish_name, \
                 price, \
                 description, \
                 category, \
                 spice_level, \
                 case \
                     when spice_level = 0 then '不辣' \
                     when spice_level = 1 then '微辣' \
                     when spice_level = 2 then '中辣' \
                     when spice_level = 3 then '重辣' \
                     else '无辣' \
                     end as spice_text, \
                 flavor, \
                 main_ingredients, \
                 cooking_method, \
                 is_vegetarian, \
                 allergens, \
                 is_available, \
                 is_featured
          from menu_items
          where is_available = 1 \
          """
    with pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"), ) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            item_list = []
            for item in result:
                item_list.append(MenuItem(
                    id=item['id'],
                    dish_name=item['dish_name'],
                    price=item['price'],
                    formatted_price=f"￥{item['price']:.2f}",
                    description=item['description'],
                    category=item['category'],
                    spice_level=item['spice_level'],
                    spice_text=item['spice_text'],
                    flavor=item['flavor'],
                    main_ingredients=item['main_ingredients'],
                    cooking_method=item['cooking_method'],
                    is_vegetarian=item['is_vegetarian'],
                    allergens=item['allergens'],
                    is_available=item['is_available']
                ))
            return MenuListResponse(
                success=True,
                menu_items=item_list,
                count=len(item_list),
                message="success"
            )


@app.post("/delivery", response_model=DeliveryResponse)
async def delivery(request: DeliveryRequest):
    """配置接口查询"""
    result = await get_delivery_info(request.address, request.travel_mode)
    return result
