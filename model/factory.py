import os
from functools import lru_cache

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from utils.config_handler import rag_conf


class ChatModelFactory:
    """推理模型工厂: 按 chat_provider 配置切换提供方
    - dashscope: 通义千问 (ChatQwen, OpenAI兼容接口)
    - kimi: 月之暗面 Kimi (ChatOpenAI 指向 Moonshot OpenAI兼容接口)
    streaming=True 统一开启, 配合 stream_mode='messages' 逐token推送
    """
    PROVIDERS = {
        'dashscope': {
            'model_key': 'dashscope_chat_model',
            'env_key': 'DASHSCOPE_API_KEY',
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        },
        'kimi': {
            'model_key': 'kimi_chat_model',
            'env_key': 'MOONSHOT_API_KEY',
            # 按 key 前缀自动选择端点:
            # sk-kimi- 开头 → Kimi Code 订阅套餐(coding 专用端点)
            # 普通 sk- 开头 → Moonshot 开放平台(按量付费)
            'base_url': None,  # 在 generator 中按 key 前缀决定
        },
    }

    def generator(self):
        provider = rag_conf.get('chat_provider', 'dashscope')
        conf = self.PROVIDERS.get(provider)
        if conf is None:
            raise ValueError(f'[ChatModelFactory]未知的chat_provider: {provider}, 可选: {list(self.PROVIDERS)}')

        model_name = rag_conf[conf['model_key']]
        api_key = os.environ.get(conf['env_key'])
        if not api_key:
            raise ValueError(f'[ChatModelFactory]使用{provider}需在.env中配置{conf["env_key"]}')

        # 统一容错参数: 超时 + 自动重试, 避免网络抖动直接终止整条流水线
        resilience = {
            'timeout': rag_conf.get('llm_timeout', 60),
            'max_retries': rag_conf.get('llm_max_retries', 2),
        }

        if provider == 'dashscope':
            # 使用ChatQwen, 解决ChatTongyi流式+工具调用arguments格式报错的问题
            return ChatQwen(
                model=model_name,
                api_key=api_key,
                base_url=conf['base_url'],
                streaming=True,
                **resilience,
            )
        # kimi 及其他 OpenAI 兼容提供方
        base_url = conf['base_url']
        if provider == 'kimi' and base_url is None:
            base_url = ('https://api.kimi.com/coding/v1' if api_key.startswith('sk-kimi-')
                        else 'https://api.moonshot.cn/v1')
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            streaming=True,
            **resilience,
        )


class EmbeddingsFactory:
    def generator(self):
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 懒加载单例: import 本模块不再要求 API key 就绪, 首次使用时才实例化
@lru_cache(maxsize=1)
def get_chat_model():
    return ChatModelFactory().generator()


@lru_cache(maxsize=1)
def get_embed_model():
    return EmbeddingsFactory().generator()


def __getattr__(name: str):
    """兼容旧代码的 from model.factory import chat_model / embed_model (PEP 562)"""
    if name == 'chat_model':
        return get_chat_model()
    if name == 'embed_model':
        return get_embed_model()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
