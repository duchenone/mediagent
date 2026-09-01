"""检索消融实验: 拆开"召回预算"与"检索方法"的贡献

四个臂(均全库无过滤检索):
    1. 向量 top-3          —— 优化前线上配置(基线)
    2. 向量 top-6          —— 只扩大预算, 检验预算的贡献
    3. 混合 top-6(无rerank) —— 只加BM25混合召回, 检验召回侧的贡献
    4. 混合+rerank top-6   —— 完整链路, 检验rerank的贡献

用法: python tests/run_ablation.py [--filter]  (默认全库无过滤; --filter 按科室过滤)
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.rag_service import RagSummarizeService  # noqa: E402
from utils.path_tool import get_abs_path  # noqa: E402


def is_hit(docs, keywords) -> bool:
    text = '\n'.join(d.page_content for d in docs)
    return any(kw in text for kw in keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', action='store_true', help='按科室过滤检索(默认全库)')
    args = parser.parse_args()

    cases = json.load(open(get_abs_path('tests/eval_dataset.json'), encoding='utf-8'))
    rag = RagSummarizeService()
    scope = '按科室过滤' if args.filter else '全库无过滤'

    arms = {name: {'hits': 0, 'fails': []} for name in
            ['向量top-3(基线)', '向量top-6', '混合top-6(无rerank)', '混合+rerank top-6']}

    for c in cases:
        dept = c['department'] if args.filter else None
        q, kws = c['query'], c['expect']
        results = {
            '向量top-3(基线)': rag.vector_store.get_retriever(dept, k=3).invoke(q),
            '向量top-6': rag.vector_store.get_retriever(dept, k=6).invoke(q),
            '混合top-6(无rerank)': rag.vector_store.get_hybrid_retriever(dept).invoke(q)[:6],
            '混合+rerank top-6': rag.retriever_docs(q, dept),
        }
        for name, docs in results.items():
            if is_hit(docs, kws):
                arms[name]['hits'] += 1
            else:
                arms[name]['fails'].append(c['id'])

    n = len(cases)
    print(f'\n检索消融实验 | 用例{n}条 | 检索范围: {scope}\n')
    print(f'{"方案":<22}{"命中率":<14}未命中用例')
    for name, a in arms.items():
        rate = round(a['hits'] / n * 100, 1)
        print(f"{name:<22}{a['hits']}/{n} ({rate}%){'':<4}{a['fails'] or '-'}")
    print('\n解读: 向量top-6 vs top-3 → 预算贡献; '
          '混合无rerank vs 混合+rerank → rerank贡献(等预算)')


if __name__ == '__main__':
    main()
