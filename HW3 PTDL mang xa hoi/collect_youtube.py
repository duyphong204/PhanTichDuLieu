# collect_youtube.py
from googleapiclient.discovery import build
import pandas as pd
import time

API_KEY = "AIzaSyBsZEdZ0NdLiAiyKpOsZtxXwjrAEA3GL8U"
youtube = build('youtube', 'v3', developerKey=API_KEY)


# ============================================
# HÀM LẤY COMMENT 1 VIDEO
# ============================================
def get_comments(video_id, max_results=200):
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        req = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        res = req.execute()

        for item in res['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']
            author = top_comment.get('authorDisplayName')

            # top level
            comments.append({
                'video_id': video_id,
                'comment_id': item['id'],
                'author': author,
                'parent': None
            })

            # replies
            if 'replies' in item:
                for reply in item['replies']['comments']:
                    r = reply['snippet']
                    comments.append({
                        'video_id': video_id,
                        'comment_id': reply['id'],
                        'author': r.get('authorDisplayName'),
                        'parent': author
                    })

        next_page_token = res.get('nextPageToken')
        if not next_page_token:
            break

        time.sleep(0.1)

    return comments


# ============================================
# DANH SÁCH VIDEO THU THẬP
# ============================================
video_ids = [
    'pHSb00vip60',
    'ThpomNoD5c4',
    '52Wg5S_LxtA',
    'y8-hiK1jNW8',
    'xzUzMpttMqU',
    'lSgXK_D5AB8',
    'S4tFO4lCfjk',
    'qgOtjcsYZ2Q',
    'EEUAJM68gDE',
    'JlCJxQhrzVg',

]

all_comments = []
for vid in video_ids:
    print(f"Crawling video {vid} ...")
    all_comments += get_comments(vid, max_results=2000)

df = pd.DataFrame(all_comments)

# LƯU RAW
df.to_csv('data/youtube_comments.csv', index=False)
print("Saved youtube_comments.csv")


# ============================================
# TẠO EDGE LIST
# ============================================
edges = df[df['parent'].notnull()][['author', 'parent']]
edges = edges.rename(columns={'author': 'source', 'parent': 'target'})

# XÓA SELF LOOP
edges = edges[edges['source'] != edges['target']]

# TÍNH TRỌNG SỐ CẠNH
edges['weight'] = 1
edges = edges.groupby(['source', 'target'], as_index=False).agg({'weight': 'sum'})
edges.to_csv('data/edges.csv', index=False)

print("Saved edges.csv")


# ============================================
# TẠO NODES LIST
# ============================================
all_nodes = pd.unique(
    df['author'].dropna().tolist() +
    df['parent'].dropna().tolist()
)

nodes = pd.DataFrame({'id': all_nodes})
nodes.to_csv("data/nodes.csv", index=False)

print("Saved nodes.csv")
print("DONE ✓")
