from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


def print_prompt(prompt):
    print('*'*20)
    print(prompt.to_string())
    print('*' * 20)
    return prompt

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self,query: str,department: str | list[str] | None = None) -> list[Document]:
        """检索参考文档,可按专科(department元数据)过滤,None/空则全库检索"""
        if department:
            return self.vector_store.get_retriever(department).invoke(query)
        return self.retriever.invoke(query)

    def rag_retrieve(self,query: str,department: str | list[str] | None = None) -> str:
        """轻量检索: 返回原始片段文本+真实来源, 不经过LLM总结
        供多Agent流水线使用, 由上层Agent自行消化上下文"""
        docs = self.retriever_docs(query,department)
        if not docs:
            return '知识库中未检索到相关内容'
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', '未知来源')
            dept = doc.metadata.get('department', '未知科室')
            parts.append(f'【资料{i}|来源: {source}({dept})】\n{doc.page_content}')
        return '\n\n'.join(parts)

    def rag_summarize(self,query: str,department: str | list[str] | None = None) -> str:
        context_docs = self.retriever_docs(query,department)
        context = ''
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f'[参考资料{counter} : 参考资料: {doc.page_content} | 参考元数据: {doc.metadata}\n]'

        return self.chain.invoke(
            {
                'input':query,
                'context':context,
            }
        )

if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_summarize('高血压的诊断标准是什么'))
