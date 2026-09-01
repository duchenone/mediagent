"""RAG 检索质量评测脚本

用法:
    python tests/run_eval.py              # 评测当前配置(混合检索+rerank, 带科室过滤)
    python tests/run_eval.py --compare    # 同时跑 朴素向量基线 vs 当前配置, 输出对比
    python tests/run_eval.py --compare --no-filter
                                          # 全库检索(不带科室过滤), 搜索空间放大8倍,
                                          # 检验混合检索+rerank在开放检索下的真实增益

指标: 检索命中率 —— 召回分片中包含任一期望关键词即记为命中。
评测报告写入 tests/eval_report.md, 供简历/面试引用真实数据。
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.rag_service import RagSummarizeService  # noqa: E402
from utils.config_handler import chroma_conf  # noqa: E402
from utils.path_tool import get_abs_path  # noqa: E402


def run_round(service: RagSummarizeService, cases: list[dict], enhanced: bool,
              department_filter: bool) -> list[dict]:
    """跑一轮评测, 返回每条用例的结果
    department_filter=False 时全库检索, 科室推断错误/过滤失效的场景由此暴露"""
    results = []
    for case in cases:
        dept = case['department'] if department_filter else None
        start = time.time()
        if enhanced:
            docs = service.retriever_docs(case['query'], dept)
        else:
            # 基线: 优化前配置 —— 朴素向量 top-3 检索(无混合召回、无重排)
            docs = service.vector_store.get_retriever(dept, k=3).invoke(case['query'])
        latency = time.time() - start

        text = '\n'.join(d.page_content for d in docs)
        hit = any(kw in text for kw in case['expect'])
        results.append({
            **case,
            'hit': hit,
            'latency': round(latency, 2),
            'n_docs': len(docs),
            'top_score': docs[0].metadata.get('relevance_score') if docs else None,
        })
    return results


def summarize(results: list[dict]) -> dict:
    hits = sum(r['hit'] for r in results)
    by_type = {}
    for r in results:
        t = r.get('type', 'standard')
        by_type.setdefault(t, {'total': 0, 'hits': 0})
        by_type[t]['total'] += 1
        by_type[t]['hits'] += r['hit']
    return {
        'total': len(results),
        'hits': hits,
        'hit_rate': round(hits / len(results) * 100, 1),
        'avg_latency': round(sum(r['latency'] for r in results) / len(results), 2),
        'by_type': {
            t: f"{v['hits']}/{v['total']} ({round(v['hits'] / v['total'] * 100, 1)}%)"
            for t, v in sorted(by_type.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true', help='与朴素向量检索基线对比')
    parser.add_argument('--no-filter', action='store_true',
                        help='全库检索(不带科室过滤), 暴露开放检索下的真实差距')
    args = parser.parse_args()
    dept_filter = not args.no_filter

    with open(get_abs_path('tests/eval_dataset.json'), 'r', encoding='utf-8') as f:
        cases = json.load(f)

    service = RagSummarizeService()
    scope = '按科室过滤' if dept_filter else '全库(无科室过滤)'
    print(f'评测用例: {len(cases)}条 | 检索范围: {scope} | 当前配置: k={chroma_conf["k"]}, '
          f'fetch_k={chroma_conf.get("fetch_k")}, '
          f'hybrid={chroma_conf.get("enable_hybrid")}, rerank={chroma_conf.get("enable_rerank")}')

    enhanced = run_round(service, cases, enhanced=True, department_filter=dept_filter)
    enhanced_summary = summarize(enhanced)

    baseline, baseline_summary = None, None
    if args.compare:
        baseline = run_round(service, cases, enhanced=False, department_filter=dept_filter)
        baseline_summary = summarize(baseline)

    # ── 控制台输出 ──
    for r in enhanced:
        mark = '✅' if r['hit'] else '❌'
        print(f"{mark} [{r['id']:>2}] {r['department']} | {r['query'][:30]} | "
              f"{r['latency']}s | top分={r['top_score']}")
    print(f"\n命中率: {enhanced_summary['hits']}/{enhanced_summary['total']} "
          f"= {enhanced_summary['hit_rate']}% | 平均延迟 {enhanced_summary['avg_latency']}s "
          f"| 分难度: {enhanced_summary['by_type']}")
    if baseline_summary:
        print(f"基线(朴素向量)命中率: {baseline_summary['hits']}/{baseline_summary['total']} "
              f"= {baseline_summary['hit_rate']}% | 平均延迟 {baseline_summary['avg_latency']}s "
              f"| 分难度: {baseline_summary['by_type']}")

    # ── 落盘报告 ──
    lines = [
        '# RAG 检索质量评测报告',
        f"\n- 评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 用例数: {len(cases)}",
        f"- 检索范围: {scope}",
        f"- 配置: k={chroma_conf['k']}, fetch_k={chroma_conf.get('fetch_k')}, "
        f"hybrid={chroma_conf.get('enable_hybrid')}, rerank={chroma_conf.get('enable_rerank')}, "
        f"rerank_model={chroma_conf.get('rerank_model')}",
        f"\n## 总体指标\n",
        f"| 方案 | 命中率 | 平均延迟 | 分难度命中率 |",
        f'|---|---|---|---|',
        f"| 混合检索+rerank | {enhanced_summary['hit_rate']}% ({enhanced_summary['hits']}/{enhanced_summary['total']}) | {enhanced_summary['avg_latency']}s | {enhanced_summary['by_type']} |",
    ]
    if baseline_summary:
        lines.append(
            f"| 朴素向量top-k(基线) | {baseline_summary['hit_rate']}% ({baseline_summary['hits']}/{baseline_summary['total']}) | {baseline_summary['avg_latency']}s | {baseline_summary['by_type']} |"
        )
    lines += ['\n## 逐条明细(混合检索+rerank)\n',
              '| ID | 难度 | 科室 | 查询 | 命中 | 延迟 | top相关度 |',
              '|---|---|---|---|---|---|---|']
    for r in enhanced:
        lines.append(f"| {r['id']} | {r.get('type', 'standard')} | {r['department']} | {r['query']} | "
                     f"{'✅' if r['hit'] else '❌'} | {r['latency']}s | {r['top_score']} |")

    report_name = 'tests/eval_report.md' if dept_filter else 'tests/eval_report_nofilter.md'
    report_path = Path(get_abs_path(report_name))
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n报告已写入: {report_path}')


if __name__ == '__main__':
    main()
