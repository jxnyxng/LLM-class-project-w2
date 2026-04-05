from vectorstore import init_vectorstore, load_vector_from_local
from chains import build_rag_chain

init_vectorstore()

vectorstore = load_vector_from_local()
chain = build_rag_chain(vectorstore=vectorstore)

print("========== [1. 에러 요약 리포트] ==========")
q1 = "Tool Choice 에러가 자꾸 나는데, 원인이랑 최종 해결책을 하나의 리포트로 요약해 줘."
res1 = chain.invoke(q1)
print(res1)
print("\n")


print("========== [2. 통합 체크리스트] ==========")
q2 = "PostgresSaver를 도입하려고 하는데, 연관된 DB 설정 이슈들 참고해서 주의해야 할 통합 체크리스트 좀 만들어 줘."
res2 = chain.invoke(q2)
print(res2)
print("\n")


print("========== [3. 코드 버그 리뷰] ==========")
q3 = """아래 코드 좀 봐줄래? 잠재적 버그 지적해주고 수정된 코드 스니펫 제안해 줘.

def setup_database():
    db = PostgresSaver()
    db.connect()
    return db
"""
res3 = chain.invoke(q3)
print(res3)
print("\n")