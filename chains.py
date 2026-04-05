from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.documents.base import Document


SYSTEM_PROMPT = """당신은 시니어 개발자이자 에러 해결 AI 어시스턴트입니다.
사용자의 질문이나 코드에 대해 정확하게 답변하세요.

규칙:
1. 반드시 아래 제공된 [참고 문서]의 내용을 바탕으로 답변하세요.
2. 에러 메시지(예: Tool Choice 에러)에 대한 질문이면, 참고 문서의 이슈와 댓글을 종합하여 '원인'과 '최종 해결책'을 하나의 리포트로 요약하세요.
3. 특정 기술(예: PostgresSaver)이나 DB 설정 이슈에 대한 질문이면, 참고 문서를 바탕으로 기술 도입 시 주의해야 할 '통합 체크리스트'를 마크다운 형식으로 생성하세요.
4. 사용자가 코드를 입력하면, 참고 문서를 바탕으로 "현재 코드의 잠재적 버그"를 지적하고, 수정된 코드 스니펫을 제안하세요.

[참고 문서]
{context}"""

def build_rag_chain(vectorstore):
    load_dotenv()

    llm = ChatGroq(
        model="llama-3.1-8b-instant"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs= {
            "k": 5
        }
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])


    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def format_docs(docs: list[Document]) -> str:

    if not docs:
        return "관련 문서를 찾지 못했습니다."

    sections = []

    for i, doc in enumerate(docs, 1):
        doc_id = doc.metadata.get("id", "알수없음")
        doc_type = doc.metadata.get("type", "일반")
        title = doc.metadata.get("title", "제목없음")
        
        section = f"{i}. [ID: {doc_id}] 타입: {doc_type} | 제목: {title}"
        section += f"\n내용: {doc.page_content}"
        sections.append(section)

    return "\n\n".join(sections)