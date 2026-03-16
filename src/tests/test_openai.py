from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
res = llm.invoke("Xin chào bạn là ai")
print(res.content)