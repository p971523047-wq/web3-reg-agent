"""
本模块用于抓取香港证监会（SFC）和美国 SEC 的政策内容，进行基础清洗、结构化为 JSON，
并可选地通过 Jina 向量接口做简单深度检索。
运行前请确认已安装依赖：requests、beautifulsoup4。
如需启用 Jina 检索，请在环境变量中提供 JINA_API_KEY。
"""

import json  # 用于存储清洗后的政策数据为 JSON
import os  # 用于读取环境变量（如 JINA_API_KEY）
import re  # 用于简单的正则清洗文本和抽取日期
from typing import Dict, List  # 类型注解，帮助初学者理解数据结构
from urllib.parse import urljoin  # 把相对链接转成绝对链接，防止抓取失败

import requests  # 发送 HTTP 请求抓取网页内容
from bs4 import BeautifulSoup  # 解析 HTML 结构，提取需要的文本和链接


# ======================== 抓取相关模块 ======================== #
def fetch_html(url: str, timeout: int = 10) -> str:
    """抓取单个网页的 HTML。"""
    headers = {"User-Agent": "Mozilla/5.0 (PolicyCrawler/1.0)"}  # 伪装浏览器头，减少被拦截概率
    resp = requests.get(url, headers=headers, timeout=timeout)  # 发送 GET 请求
    resp.raise_for_status()  # 如果状态码非 200，则抛出异常便于排查
    return resp.text  # 返回 HTML 文本


def extract_links_by_keywords(soup: BeautifulSoup, base_url: str, keywords: List[str]) -> List[Dict]:
    """从页面中找出含关键词的链接（标题或 URL）。"""
    results = []  # 存放符合条件的链接列表
    for a in soup.find_all("a"):  # 遍历页面上的所有超链接
        title = (a.get_text() or "").strip()  # 获取链接文字并去掉首尾空白
        href = a.get("href")  # 获取超链接地址
        if not href:  # 若没有链接则跳过
            continue  # 继续下一个循环
        full_url = urljoin(base_url, href)  # 将相对路径转为绝对 URL
        text_to_check = f"{title} {full_url}".lower()  # 将标题和 URL 合并转小写，便于关键词匹配
        if any(kw.lower() in text_to_check for kw in keywords):  # 如果包含任一关键词
            results.append({"title": title, "url": full_url})  # 加入结果列表
    return results  # 返回筛选出的链接


def fetch_and_extract(base_url: str, entry_paths: List[str], keywords: List[str]) -> List[Dict]:
    """从入口页开始抓取，找到包含关键词的链接并拉取其正文。"""
    records = []  # 存储每条政策记录
    for path in entry_paths:  # 遍历入口路径
        entry_url = urljoin(base_url, path)  # 组合入口完整 URL
        try:
            html = fetch_html(entry_url)  # 抓取入口页 HTML
        except Exception as e:  # 捕获网络异常避免程序中断
            print(f"[warn] 抓取入口页失败 {entry_url}: {e}")  # 打印警告
            continue  # 继续处理下一个入口
        soup = BeautifulSoup(html, "html.parser")  # 解析入口页 HTML
        links = extract_links_by_keywords(soup, base_url, keywords)  # 按关键词过滤链接
        for link in links:  # 遍历符合条件的链接
            try:
                page_html = fetch_html(link["url"])  # 抓取链接对应页面
            except Exception as e:  # 捕获异常
                print(f"[warn] 抓取详情页失败 {link['url']}: {e}")  # 打印警告
                continue  # 跳过该链接
            page_soup = BeautifulSoup(page_html, "html.parser")  # 解析详情页
            text = page_soup.get_text(separator="\n")  # 把页面所有可见文字拼成一段文本
            cleaned = clean_text(text)  # 调用清洗函数去掉噪声
            records.append(  # 追加到记录列表
                {
                    "title": link["title"],
                    "url": link["url"],
                    "source": base_url,
                    "raw_text": text,
                    "clean_text": cleaned,
                    "date": extract_date(cleaned),
                }
            )
    return records  # 返回抓取的记录


# ======================== 文本清洗与结构化模块 ======================== #
def clean_text(text: str) -> str:
    """基础清洗：移除多余空白和重复换行。"""
    text = re.sub(r"\s+", " ", text)  # 用一个空格替换多余空白
    text = text.strip()  # 去掉首尾空白
    return text  # 返回清洗后的文本


def extract_date(text: str) -> str:
    """简单提取日期（格式 YYYY-MM-DD 或 YYYY/MM/DD），找不到则返回空字符串。"""
    m = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", text)  # 使用正则匹配常见日期格式
    return m.group(0) if m else ""  # 找到则返回日期，否则返回空字符串


def save_json(data: List[Dict], path: str) -> None:
    """把抓取结果保存为 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:  # 以 UTF-8 编码写文件
        json.dump(data, f, ensure_ascii=False, indent=2)  # 写入并保持中文


# ======================== Jina 深度检索模块 ======================== #
JINA_API_URL = "https://api.jina.ai/v1/embeddings"  # 官方 Jina 嵌入 API 地址


def embed_with_jina(text: str, jina_api_key: str) -> List[float]:
    """调用 Jina 嵌入接口获取向量。"""
    payload = {"input": text, "model": "jina-embeddings-v2-base-en"}  # 指定模型与输入
    headers = {"Authorization": f"Bearer {jina_api_key}"}  # 在请求头里放入密钥
    resp = requests.post(JINA_API_URL, headers=headers, json=payload, timeout=15)  # 发送 POST 请求
    resp.raise_for_status()  # 如果失败抛异常便于排查
    data = resp.json()  # 解析返回 JSON
    return data["data"][0]["embedding"]  # 提取向量数组


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """手工计算余弦相似度。"""
    if not vec_a or not vec_b:  # 若任意向量为空，返回 0
        return 0.0  # 返回 0 表示无相似度
    dot = sum(a * b for a, b in zip(vec_a, vec_b))  # 计算点积
    norm_a = sum(a * a for a in vec_a) ** 0.5  # 计算向量 A 的模
    norm_b = sum(b * b for b in vec_b) ** 0.5  # 计算向量 B 的模
    if norm_a == 0 or norm_b == 0:  # 防止除零
        return 0.0  # 若任一模长为 0，返回 0
    return dot / (norm_a * norm_b)  # 返回余弦相似度


def build_embeddings(records: List[Dict], jina_api_key: str) -> None:
    """为每条记录生成并存储向量，原地更新 records。"""
    for rec in records:  # 遍历每条记录
        try:
            rec["embedding"] = embed_with_jina(rec["clean_text"], jina_api_key)  # 生成并保存向量
        except Exception as e:  # 捕获异常
            print(f"[warn] Jina 向量生成失败 {rec.get('url')}: {e}")  # 打印警告
            rec["embedding"] = []  # 失败则留空，便于后续判断


def search_with_jina(records: List[Dict], query: str, jina_api_key: str, top_k: int = 5) -> List[Dict]:
    """对抓取结果做向量检索，返回相似度最高的记录。"""
    query_vec = embed_with_jina(query, jina_api_key)  # 为查询生成向量
    scored = []  # 存放打分结果
    for rec in records:  # 遍历记录
        sim = cosine_similarity(query_vec, rec.get("embedding", []))  # 计算相似度
        scored.append({**rec, "score": sim})  # 记录相似度分数
    scored.sort(key=lambda x: x["score"], reverse=True)  # 按分数从高到低排序
    return scored[:top_k]  # 返回前 top_k 条


def keyword_fallback(records: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
    """如果没有 Jina 密钥，则用简单关键词匹配作为兜底。"""
    q = query.lower()  # 把查询转成小写
    scored = []  # 存放打分结果
    for rec in records:  # 遍历记录
        text = rec.get("clean_text", "").lower()  # 取出清洗文本并转小写
        score = text.count(q)  # 简单统计关键词出现次数作为得分
        scored.append({**rec, "score": score})  # 保存得分
    scored.sort(key=lambda x: x["score"], reverse=True)  # 按分数排序
    return scored[:top_k]  # 返回前 top_k 条


# ======================== 主流程控制模块 ======================== #
def main():
    """统一 orchestrator：抓取、清洗、保存并检索。"""
    keywords = ["web3", "virtual asset", "虚拟资产", "crypto", "stablecoin", "数字资产"]  # 关键词列表

    # 定义 SFC 与 SEC 入口路径，方便日后扩展
    sfc_base = "https://www.sfc.hk"  # SFC 主域
    sfc_entries = ["/en/News-and-announcements/Policy-statements", "/en/News-and-announcements/Announcements"]  # 常见政策和公告入口

    sec_base = "https://www.sec.gov"  # SEC 主域
    sec_entries = ["/news/pressreleases", "/news/public-statements"]  # 常见新闻稿和公开声明入口

    # 抓取两个站点的政策
    print("[info] 开始抓取 SFC ...")  # 打印进度
    sfc_records = fetch_and_extract(sfc_base, sfc_entries, keywords)  # 抓取 SFC

    print("[info] 开始抓取 SEC ...")  # 打印进度
    sec_records = fetch_and_extract(sec_base, sec_entries, keywords)  # 抓取 SEC

    # 合并所有记录
    all_records = sfc_records + sec_records  # 将两处数据合并
    print(f"[info] 共获取 {len(all_records)} 条候选记录")  # 打印总数

    # 保存清洗结果
    os.makedirs("output", exist_ok=True)  # 确保 output 目录存在
    save_json(all_records, "output/policies.json")  # 保存 JSON
    print("[info] 已写入 output/policies.json")  # 提示写入完成

    # 检索示例（可选）
    jina_api_key = os.getenv("JINA_API_KEY", "")  # 从环境变量读取 Jina 密钥
    query = "禁止 稳定币"  # 示例查询关键词，可按需修改
    if jina_api_key:  # 如果提供了密钥
        print("[info] 检测到 JINA_API_KEY，使用 Jina 深度检索")  # 打印提示
        build_embeddings(all_records, jina_api_key)  # 为每条记录生成向量
        top_hits = search_with_jina(all_records, query, jina_api_key, top_k=5)  # 做向量检索
    else:
        print("[info] 未检测到 JINA_API_KEY，使用关键词匹配兜底检索")  # 打印提示
        top_hits = keyword_fallback(all_records, query, top_k=5)  # 使用兜底检索

    # 打印检索结果概览
    print("[info] 检索结果（前 5 条）：")  # 提示
    for item in top_hits:  # 遍历结果
        print(f"- {item.get('title')} | {item.get('url')} | score={item.get('score', 0):.4f}")  # 打印标题、链接与得分


if __name__ == "__main__":  # 确保作为脚本运行时才执行 main
    main()  # 运行主流程
