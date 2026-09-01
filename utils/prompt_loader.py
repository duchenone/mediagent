from functools import lru_cache

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


@lru_cache(maxsize=None)
def _load_prompt(config_key: str) -> str:
    """按 prompts.yml 配置项读取提示词文件, 进程内缓存(避免每次LLM调用重读磁盘).
    提示词文件变更后需重启进程或调用 _load_prompt.cache_clear() 生效"""
    try:
        path = get_abs_path(prompts_conf[config_key])
    except KeyError as e:
        logger.error(f'[prompt_loader]yaml中缺少配置项: {config_key}')
        raise e
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f'[prompt_loader]读取提示词{path}出错: {str(e)}')
        raise e


def load_system_prompts() -> str:
    return _load_prompt('main_prompt_path')


def load_rag_prompts() -> str:
    return _load_prompt('rag_summarize_prompt_path')


def load_report_prompts() -> str:
    return _load_prompt('report_prompt_path')


def load_pipeline_stage_prompt(stage: int) -> str:
    """加载流水线阶段 prompt (stage: 1-4)"""
    return _load_prompt(f'pipeline_stage{stage}_prompt_path')


if __name__ == '__main__':
    print(load_report_prompts())
