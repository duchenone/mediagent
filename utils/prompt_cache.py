"""Kimi Context Caching 客户端 (实验性, 默认关闭)

原理: 将各阶段固定不变的 system prompt 通过 POST /v1/caching 注册为缓存,
后续 chat 请求在 messages 开头放 {"role": "cache", "content": "cache_id=xxx"}
引用缓存内容, 命中缓存的 token 按低价计费, 可显著降低流水线重复开销。

注意:
- 仅在 chat_provider=kimi 且 pipeline.yml 中 enable_prompt_cache=true 时启用
- 创建失败/调用失败均静默降级为不缓存, 不影响主流程
- cache TTL 到期自动失效, 无需手动清理
"""

import json
import os
import urllib.request

from utils.logger_handler import logger

CACHE_API = 'https://api.moonshot.cn/v1/caching'


def create_prompt_cache(name: str, system_prompt: str, model: str, ttl: int = 3600) -> str | None:
    """注册一段 system prompt 为 Kimi 上下文缓存, 返回 cache_id; 失败返回 None"""
    api_key = os.environ.get('MOONSHOT_API_KEY', '')
    if not api_key or api_key.startswith('sk-请填入'):
        logger.warning('[PromptCache] 未配置有效 MOONSHOT_API_KEY, 跳过缓存注册')
        return None
    try:
        body = json.dumps({
            'model': model,
            'name': name,
            'description': f'mediagent pipeline stage prompt: {name}',
            'messages': [{'role': 'system', 'content': system_prompt}],
            'ttl': ttl,
        }).encode('utf-8')
        req = urllib.request.Request(
            CACHE_API, data=body,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        cache_id = data.get('id')
        if cache_id:
            logger.info(f'[PromptCache] {name} 缓存注册成功: {cache_id}')
            return cache_id
        logger.warning(f'[PromptCache] {name} 注册返回无id: {data}')
        return None
    except Exception as e:
        logger.warning(f'[PromptCache] {name} 注册失败, 降级为不缓存: {e}')
        return None


def cache_message(cache_id: str) -> dict:
    """生成引用缓存的 message dict, 置于 messages 开头"""
    return {'role': 'cache', 'content': f'cache_id={cache_id}'}
