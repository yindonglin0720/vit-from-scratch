"""评分方式升级 —— D2：exact_match / keyword_match / human_score"""
import json
import re

# ============================================
# 🔴 必须手敲 第1段：三层评分函数
# ============================================
def exact_match(reference, model_answer):
    """参考答案是否完整出现在模型回答中（第三周的老方法）"""
    return reference in model_answer

def keyword_match(reference, model_answer):
    """参考答案的关键词是否出现在模型回答中"""
    ref_clean = re.sub(r'[，。！？、\s]', '', reference)
    ans_clean = re.sub(r'[，。！？、\s]', '', model_answer)

    # 参考答案超过2个字时，拆成首尾双字词组，至少一个命中就算对
    if len(ref_clean) > 2:
        keywords = [ref_clean[:2], ref_clean[-2:]]
    else:
        keywords = [ref_clean]

    return any(kw in ans_clean for kw in keywords)

def human_score(reference, model_answer):
    """人工评分 0/1/2"""
    print(f"\n参考答案: {reference}")
    print(f"模型回答: {model_answer[:200]}...")
    while True:
        s = input("评分 (2=完全正确, 1=部分正确, 0=完全错误): ").strip()
        if s in ('0', '1', '2'):
            return int(s)
        print("请输入 0/1/2")

# ============================================
# 第2段：读取结果并评分
# ============================================
def evaluate(results_path, mode="auto"):
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    scores = {"exact_match": 0, "keyword_match": 0, "human_score_total": 0}
    type_scores = {}

    for r in results:
        ref = r["reference_answer"]
        ans = r["model_answer"]
        qtype = r.get("type", "unknown")

        em = exact_match(ref, ans)
        km = keyword_match(ref, ans)

        scores["exact_match"] += em
        scores["keyword_match"] += km

        if mode == "human":
            hs = human_score(ref, ans)
            scores["human_score_total"] += hs
        else:
            hs = -1

        if qtype not in type_scores:
            type_scores[qtype] = {"total": 0, "exact": 0, "keyword": 0, "human_sum": 0}
        type_scores[qtype]["total"] += 1
        type_scores[qtype]["exact"] += em
        type_scores[qtype]["keyword"] += km
        if hs >= 0:
            type_scores[qtype]["human_sum"] += hs

    n = len(results)
    print(f"\n=== 总体评分 ({n} 条) ===")
    print(f"exact_match:     {scores['exact_match']}/{n} ({scores['exact_match']/n*100:.1f}%)")
    print(f"keyword_match:   {scores['keyword_match']}/{n} ({scores['keyword_match']/n*100:.1f}%)")
    if scores["human_score_total"] > 0:
        print(f"human_score:     {scores['human_score_total']}/{n*2} ({scores['human_score_total']/(n*2)*100:.1f}%)")

    print("\n=== 按类型统计 (keyword_match) ===")
    for t, d in sorted(type_scores.items()):
        print(f"  {t:15s}: {d['keyword']}/{d['total']}")

    print("\n=== keyword_match 失败的案例 ===")
    for r in results:
        if not keyword_match(r["reference_answer"], r["model_answer"]):
            print(f"  图: {r['image']} | 类型: {r.get('type','?')}")
            print(f"  问: {r['question']}")
            print(f"  参: {r['reference_answer']}")
            print(f"  答: {r['model_answer'][:150]}...")
            print()

if __name__ == "__main__":
    results_path = r"C:\Users\text\Desktop\vit-from-scratch\outputs\vqa_results.json"
    evaluate(results_path, mode="human")