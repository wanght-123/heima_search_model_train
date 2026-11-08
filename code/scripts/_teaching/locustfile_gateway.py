"""部署环境的推理压测脚本（基于 Locust）。

提供哪些功能
------------
1. 用 Locust 模拟多名用户「同时」向模型网关发 chat 请求，制造可控的并发负载。
2. 让 Locust 自动统计压测指标——每秒请求数 RPS、失败率、延迟分位数 p50/p95/p99，
   并在 Web UI（默认 8089 端口）实时展示，无需自己写线程池与统计代码。

实现步骤
--------
—— 准备 ——
1. 用几条招聘领域的真实提示词当负载样本（取自实体抽取任务，让模型真正生成 JSON，延迟才贴近线上）。
2. 从环境变量读网关鉴权 Key 与逻辑模型名；读不到就用占位默认值（与 6.3 的网关配置示例一致）。
—— 运行时 ——
3. 每个虚拟用户循环 POST /v1/chat/completions，校验返回 200 且 choices 非空，否则记为失败。
4. 用 locust 命令启动（见教案 8.3），在浏览器看 Stats 表与曲线。

注意：本脚本只定义「压什么、怎么判成功」，并发数与加压速率在启动命令 / Web UI 里指定。
Locust 是开源压测工具，会替我们做并发调度与指标统计，所以这里看不到线程池和分位数计算代码。
"""

import os

# Locust 提供：HttpUser=会发 HTTP 的虚拟用户，between=两次请求之间的等待策略，task=标记一个压测动作
from locust import HttpUser, between, task

# TODO 重点：host 只填网关「根地址」（如 http://127.0.0.1:4000），不要带 /v1；
# 完整路径 /v1/chat/completions 在下面的 task 里拼。若误带 /v1 会变成 /v1/v1/... 而 404。
# 启动时的 --host 参数会覆盖这里的默认值。
GATEWAY_HOST = os.environ.get("LOCUST_HOST", "http://127.0.0.1:4000")

# 网关对外的鉴权 Key 与逻辑模型名（与 6.3 的 litellm_config / 消费端 .env 对齐）；
# 读不到环境变量时回退占位值，便于先跑通界面、再换真实 Key。
API_KEY = os.environ.get("LLM_API_KEY", "sk-gateway-xxx")
MODEL_NAME = os.environ.get("LLM_MODEL", "qwen3-recruitment")

# 负载样本：实体抽取任务的提示词（system 给抽取规则，user 给一段职位文本）。
# 故意用真实任务而非 "hello"——模型真正生成一段 JSON，测出的延迟才贴近线上抽取场景。
PROMPTS = [
    [
        {"role": "system", "content": "你是招聘领域实体抽取助手，从职位文本中抽取工种、工作时间、任职要求、学历，只输出 JSON。"},
        {"role": "user", "content": "急聘宠物美容师，要求1年以上宠物美容经验、持宠物美容师资格证，每周工作5天每天8小时、月休六天。"},
    ],
    [
        {"role": "system", "content": "你是招聘领域实体抽取助手，从职位文本中抽取工种、工作时间、任职要求、学历，只输出 JSON。"},
        {"role": "user", "content": "招仓库分拣员，日结五百，做滴滴司机不如做仓库分拣，能适应夜班，学历不限。"},
    ],
    [
        {"role": "system", "content": "你是招聘领域实体抽取助手，从职位文本中抽取工种、工作时间、任职要求、学历，只输出 JSON。"},
        {"role": "user", "content": "诚聘腌菜厨师，日结上午班，包吃住，三年以上灶台经验优先。"},
    ],
]


class GatewayChatUser(HttpUser):
    """模拟一个并发用户：循环向网关发 chat 请求。

    需求：以受控并发度持续打网关的 chat 接口，并判定每次请求成功或失败。
    思路：
    1. host 指向网关根地址，task 里拼 /v1/chat/completions。
    2. 带上网关鉴权 Key（Bearer），与业务调用一致，否则会被网关 401 拦下。
    3. 用 catch_response 自己判定：非 200 或 choices 为空都记失败，让失败率指标准确。
    """

    host = GATEWAY_HOST
    # 两次请求间随机等 0.1~0.5 秒：等待越短、同并发下 RPS 越高，用来逼近峰值负载。
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        """每个虚拟用户启动时构造一次请求头，避免每次请求重复拼。"""
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",  # 等于网关 master_key，缺它会被 401
            "Content-Type": "application/json",
        }
        self._idx = 0  # 轮询取 PROMPTS 的下标，让不同请求用不同样本

    @task
    def chat_completions(self) -> None:
        """发一次 chat 请求并判定成败。"""
        messages = PROMPTS[self._idx % len(PROMPTS)]  # 轮询样本，覆盖不同长度与内容
        self._idx += 1
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 256,  # 限制生成长度，压测关注延迟与吞吐，不需要长输出
            "temperature": 0,   # 贪心解码，结果稳定、延迟可比
        }
        # catch_response=True：由我们自己决定这次算成功还是失败（见下面的 success / failure）。
        # name= 让 Stats 表里这一行有固定名字，而不是把整段 URL 当行名。
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=self.headers,
            name="POST /v1/chat/completions",
            catch_response=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")  # 非 200 记失败，计入失败率
                return
            try:
                body = resp.json()
            except ValueError:
                resp.failure("响应不是合法 JSON")
                return
            choices = body.get("choices") or []
            # 模型没正常生成内容也视为失败，避免「200 但空结果」被当成成功。
            if not choices or not choices[0].get("message", {}).get("content"):
                resp.failure("choices 为空或无 content")
                return
            resp.success()  # 显式标记成功
