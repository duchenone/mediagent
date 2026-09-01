"""
yaml 配置加载: 进程内缓存, 重复调用(load_pipeline_config 曾被加载3次)不再重复读盘
k: v
"""
from functools import lru_cache

import yaml
from utils.path_tool import get_abs_path


@lru_cache(maxsize=None)
def _load_yaml(config_path: str) -> dict:
    with open(get_abs_path(config_path), "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_rag_config():
    return _load_yaml("config/rag.yml")

def load_chroma_config():
    return _load_yaml("config/chroma.yml")

def load_prompts_config():
    return _load_yaml("config/prompts.yml")

def load_agent_config():
    return _load_yaml("config/agent.yml")

def load_pipeline_config():
    return _load_yaml("config/pipeline.yml")


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
agent_conf = load_agent_config()
prompts_conf = load_prompts_config()

if __name__ == '__main__':
    print(rag_conf["embedding_model_name"])
