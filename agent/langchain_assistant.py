import json
import time

from dotenv import load_dotenv
from pathlib import  Path

from langchain.agents.middleware import ToolCallLimitMiddleware, ToolRetryMiddleware, ModelRetryMiddleware, \
    ModelCallLimitMiddleware
from langchain.tools import tool
import pymysql
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field
from pymysql.cursors import DictCursor
import os
import sys
ROOT_PATH= Path(__file__).parent.parent #获取项目根目录
sys.path.insert(0, str(ROOT_PATH)) # 将项目的根目录添加到系统环境变量中
from utils.milvus_utils import get_milvus_client, get_embedding

load_dotenv()

agent = None
_mcp_tools = None

async def get_mcp_tools():
    global _mcp_tools
    if _mcp_tools is None:
        client = MultiServerMCPClient(
            {
                "amap-maps":{
                    "transport":"streamable-http",
                    "url":"https://mcp.amap.com/mcp?key=a2b269d20ffa4351a3d397eed653e269"
                }
        })
        _mcp_tools =await client.get_tools()
    return _mcp_tools

class ReservationToolArgsInfo(BaseModel):
    num_people:int = Field(description="预订人数")
    num_children:int = Field(description="预订儿童人数")
    arrival_time:str = Field(description="到店时间，格式：YYYY-MM-DD HH")
    seat_preference:str= Field(description="座位偏好")
    main_dish_preference:str=Field(description="主菜偏好")
    other_comments:str= Field(description="其他备注")

#1.agent能够推荐特色菜
@tool
def search_featured_dish():
    """用来搜索餐厅的特色菜"""
    sql = """
    select
        menu_items.dish_name as 菜品名称,
        menu_items.price as 价格,
        menu_items.category as 菜品类别,
        menu_items.description as 描述,
        menu_items.spice_level as 麻辣程度,
        menu_items.flavor as 口味,
        menu_items.main_ingredients as 主料,
        menu_items.cooking_method as 烹饪方法,
        menu_items.is_vegetarian as 是否素食,
        menu_items.allergens as 过敏源
        from menu_items
    where is_featured=1
    """
    with pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            return result

# 2.agent能够推荐菜品
@tool
def user_favorite_dishes(query:str):
    """根据用户的口味要求推荐菜品"""
    # 创建Milvus客户端连接
    client = get_milvus_client()

    # 初始化HuggingFace文本嵌入模型
    embedding = get_embedding()

    # 1. 生成query的向量
    vector_query = embedding.embed_query(query)

    # 2. 相似度检索:语义层面进行相似度检索
    search_res = client.search(
        collection_name=os.getenv("MILVUS_COLLECTION"),
        data=[vector_query],  # 查询向量
        anns_field="vector",  # 指定检索的字段
        output_fields=["text"],  # 指定输出的字段
        limit=3  # 指定输出的条数
    )
    print(search_res)

    # 3. 获取结果
    if search_res:
        all_results = search_res[0]
        final_result = []
        for result in all_results:
            distance = result["distance"]
            if distance > 0.5:
                text = result["entity"]["text"]
                final_result.append(text)
        return final_result
    else:
        return "当前菜单中没有找到和用户喜好匹配的菜品！"

@tool(args_schema=ReservationToolArgsInfo)
def make_reservation(num_people, num_children, arrival_time, seat_preference, main_dish_preference, other_comments):
    """餐厅座位约定"""
    try:
        sql = """
            insert into reservation_order (num_people, num_children, arrival_time, seat_preference, main_dish_preference, other_comments) values (%s, %s, %s, %s, %s, %s)
        """
        with pymysql.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE"),) as conn:
            with conn.cursor(DictCursor) as cursor:
                cursor.execute(sql, (num_people, num_children, arrival_time, seat_preference, main_dish_preference, other_comments))
                conn.commit() # 提交事务
                return "预订成功！"
    except Exception as e:
        return f"预订失败：{str(e)}"

async def get_agent():
    global agent
    if agent is None:
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver
        mcp_amap_tools = await get_mcp_tools()
        all_tools =[search_featured_dish, user_favorite_dishes, make_reservation]+ mcp_amap_tools
        checkpointer=InMemorySaver()
        # 获取系统提示词
        with open(str(ROOT_PATH / "agent"/"prompt"/"system_prompt.txt"), "r", encoding="utf-8") as f:
            system_prompt = f.read()

        agent = create_agent(
            model = "gpt-4o",
            tools= all_tools,
            system_prompt=system_prompt, #系统提示词
            checkpointer=checkpointer, #开启短期记忆
            # 工具调用限制 节约调用次数
            middleware=[
                ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
                ToolRetryMiddleware(
                    max_retries=3, # 最大重试次数
                    backoff_factor=2.0, # 重试间隔时间
                    initial_delay=1.0, # 初始重试间隔时间
                ),
                ModelRetryMiddleware(
                    max_retries=3,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                ),
                ModelCallLimitMiddleware(
                    thread_limit=50,
                    run_limit=20,
                    exit_behavior="end",
                ),
            ]

        )
    return agent


async def assistant_query(query:str, session_id:str):
    """将agent流式处理的内容，使用sse流式返回"""
    agent = await get_agent()
    config = {"configurable":{"thread_id":session_id}}
    current_data = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 1. 调用agent的流式输出
    async for (chunk, metadata) in agent.astream({
        "messages":[
                {"role":"system","content":f"你是六合餐厅的智能助手，请根据用户的输入，进行回答，或者完成预订任务,当前时间：{current_data}"},
                {"role": "user","content": query}
        ]
    },config, stream_mode="messages"):
        if type(chunk) == ToolMessage:
            continue
        #{"content":内容， "type":"token"}
        payload_str = json.dumps({"type": "token", "content": chunk.content}, ensure_ascii=False)
        yield f"data: {payload_str}\n\n" # sse有对数据的规范

async  def test_agent():
    agent = await get_agent()
    config = {"configurable":{"thread_id":"12345"}}
    current_data = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # res1 = await  agent.ainvoke({
    #     "messages":
    #         [
    #             {
    #                 "role":"system",
    #                 "content":f"你是六合餐厅的智能助手，请根据用户的输入，进行回答，或者完成预订任务,当前时间：{current_data}"
    #             },
    #             {
    #         "role": "user",
    #         "content": "请帮我约定一下明天下午4点的座位，总共4个人,其中有一位小朋友，位置最好靠窗，菜品主要偏辣一点，希望位置不要太吵闹！"
    #     }]
    # },config)
    # print(res1["messages"][-1].content)
    #
    # res2 = await  agent.ainvoke({
    #     "messages":
    #         [
    #             {
    #                 "role": "user",
    #                 "content": "好的，以上信息没问题，请帮我预定！"
    #             }]
    # }, config)
    # print(res2["messages"][-1].content)

    res1 = await  agent.ainvoke({
        "messages":
            [
                {
            "role": "user",
            "content": "我现在在中山公园，想步行过去，帮我规划路线如何去你们餐厅？"
        }]
    },config)
    print(res1["messages"][-1].content)

async def get_delivery_info(address:str, travel_mode:str):
    """根据目标地址检索配送情况"""
    # 1.创建一个agent，由agent来自动根据travel_mode来选择工具
    # 2.这个业务流程是确定，我们可以直接处理，不需要经过大模型
    all_amap_tools = await get_mcp_tools()
    tool_dict = {tool.name: tool for tool in all_amap_tools}

    origin = "六合餐厅" #计算经纬度
    address = ""#计算经纬度

    if travel_mode == "1":
        travel_mode = "骑行"
        tool_name = "maps_direction_bicycling"
        tool = tool_dict[tool_name]
        res = await tool.ainvoke({"origin":origin, "destination":address})

    elif travel_mode == "2":
        travel_mode = "驾车"
        tool_name = "maps_direction_driving"
    else:
        travel_mode = "直线"
        tool_name = "maps_distance"


    return {
        "success": True,
        "address": address,
        "distance_km": 2.5,
        "duration_min": 15,
        "in_range": True,
        "travel_mode": travel_mode,
        "message": "在配送范围内！"
    }

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
    # res = search_featured_dish.invoke(None)
    # print(res)
