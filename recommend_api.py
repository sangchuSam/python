#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# ✅ MongoDB 연결
client = MongoClient("mongodb+srv://pick:p2pZIQ4YNRegnZWk@recommend.oka4s.mongodb.net/recommend?retryWrites=true&w=majority")
db = client["recommend"]
collection = db["like"]  # ✅ 컬렉션 이름 `like`

# ✅ MongoDB 데이터 불러오기
likes = list(collection.find({}, {"_id": 0, "restaurantId": 1, "name": 1, "category": 1, "priceLevel": 1}))
df = pd.DataFrame(likes)

# ✅ 'category'와 'priceLevel'을 결합하여 특징 생성
df["features"] = df["category"] + " " + df["priceLevel"]

# ✅ TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["features"])

# ✅ 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ✅ 음식점 추천 API
@app.route("/recommend", methods=["POST"])
def recommend_restaurant():
    data = request.get_json()

    if not data:
        return jsonify({"error": "❌ JSON 데이터를 받지 못했습니다!"}), 400

    guest_id = data.get("guestId")
    preferences = data.get("preferences")

    if not guest_id or not preferences:
        return jsonify({"error": "❌ guestId 또는 preferences 값이 없습니다!"}), 400

    if preferences not in df["category"].values:
        return jsonify({"error": "❌ 해당 카테고리의 음식점이 없습니다!"}), 400

    # ✅ 입력된 카테고리와 가장 유사한 음식점 찾기
    idx = df[df["category"] == preferences].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # 자기 자신 제외, 상위 5개 추천

    # ✅ 추천 음식점 리스트 추출
    restaurant_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[restaurant_indices][["restaurantId", "name", "category", "priceLevel"]].to_dict(orient="records")

    return jsonify({"guestId": guest_id, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


