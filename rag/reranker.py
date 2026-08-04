"""检索结果重排序 (DashScope gte-rerank) + RRF 双序融合

混合检索(BM25+向量, 自身已是 RRF 融合)召回 fetch_k 条候选后的精排策略:
- 不直接采用 rerank 模型的单一排序 —— 实测发现 rerank 对"关键词型"查询
  (如药品名/检查指标)存在排序偏差, 会把精确命中的分片挤出 top-k
- 因此将 召回侧 RRF 名次 与 rerank 相关性名次 再做一次 RRF 融合:
  score(d) = 1/(60+rank_recall) + 1/(60+rank_rerank), 双排序投票, 单模型偏差不致命
- 低于 rerank_threshold 的结果被过滤, 减少无关上下文注入、抑制幻觉
- 任何异常/调用失败都降级为召回侧原始顺序截断, 不阻断检索主链路
"""

import os

import dashscope
from langchain_core.documents import Document

from utils.config_handler import chroma_conf
from utils.logger_handler import logger

# RRF 平滑常数(经典值), 防止头部结果权重过大
_RRF_K = 60


def rerank_docs(query: str, docs: list[Document], top_n: int | None = None) -> list[Document]:
    """对候选文档做 rerank + 召回顺序的 RRF 融合, 返回 top_n 条(附 relevance_score 元数据)
    docs: 混合检索召回的候选, 顺序即召回侧 RRF 名次"""
    if not docs:
        return docs
    top_n = top_n or chroma_conf['k']
    threshold = chroma_conf.get('rerank_threshold', 0.0)
    model = chroma_conf.get('rerank_model', 'gte-rerank-v2')

    try:
        resp = dashscope.TextReRank.call(
            api_key=os.environ.get('DASHSCOPE_API_KEY'),
            model=model,
            query=query,
            documents=[d.page_content for d in docs],
            top_n=len(docs),  # 取全量名次用于融合, 截断在融合后执行
            return_documents=False,
        )
        if resp.status_code != 200:
            logger.error(f'[rerank]调用失败: {resp.code} {resp.message}, 降级为召回顺序')
            return docs[:top_n]

        # RRF 融合: 召回侧名次(入参顺序) + rerank 名次
        rrf_scores: dict[int, float] = {}
        rerank_score_of: dict[int, float] = {}
        for rank, item in enumerate(resp.output.results):
            rrf_scores[item.index] = rrf_scores.get(item.index, 0.0) + 1 / (_RRF_K + rank)
            rerank_score_of[item.index] = item.relevance_score
        for rank in range(len(docs)):
            rrf_scores[rank] = rrf_scores.get(rank, 0.0) + 1 / (_RRF_K + rank)

        order = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        picked = []
        for idx in order:
            score = rerank_score_of.get(idx, 0.0)
            if score < threshold:
                continue
            doc = docs[idx]
            doc.metadata['relevance_score'] = round(score, 4)
            picked.append(doc)
            if len(picked) >= top_n:
                break

        if not picked:  # 阈值过滤后为空时兜底, 避免上层拿到空上下文
            logger.warning(f'[rerank]所有结果低于阈值{threshold}, 降级为召回顺序')
            return docs[:top_n]
        logger.info(f'[rerank]{len(docs)}条候选RRF融合后取{len(picked)}条, 最高分{picked[0].metadata["relevance_score"]}')
        return picked
    except Exception as e:
        logger.error(f'[rerank]异常: {e}, 降级为召回顺序', exc_info=True)
        return docs[:top_n]
