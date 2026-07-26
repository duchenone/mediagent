import os
from langchain_core.documents import Document
from utils.file_handler import pdf_loader,txt_loader,listdir_with_allowed_type,get_file_md5_hex
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from model.factory import embed_model


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf['collection_name'],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf['persist_directory']),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf['chunk_size'],
            chunk_overlap=chroma_conf['chunk_overlap'],
            separators=chroma_conf['separators'],
            length_function=len,
        )

    def get_retriever(self,department: str | list[str] | None = None):
        """获取检索器,可按专科(department元数据)过滤
        department: 专科名称(如'心血管内科')或专科列表,None/空则全库检索"""
        search_kwargs: dict = {'k':chroma_conf['k']}
        if department:
            departments = [department] if isinstance(department,str) else list(department)
            if len(departments) == 1:
                search_kwargs['filter'] = {'department':departments[0]}
            else:
                search_kwargs['filter'] = {'department':{'$in':departments}}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def load_document(self):
        """从数据文件夹内读取数据文件,转为向量存入向量库
        要计算文件的MD5做去重"""
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf['md5_hex_store'])):
                open(get_abs_path(chroma_conf['md5_hex_store']),'w',encoding='utf-8').close()
                return False
            with open(get_abs_path(chroma_conf['md5_hex_store']),'r',encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if md5_for_check == line:
                        return True
                return False

        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf['md5_hex_store']),'a',encoding='utf-8') as f:
                f.write(md5_for_save + '\n')

        def get_file_documents(read_path: str):
            if read_path.endswith('.pdf'):
                return pdf_loader(read_path)
            if read_path.endswith('.txt'):
                return txt_loader(read_path)
            return []

        def get_department(read_path: str):
            """按文件所在的 data 子目录推断专科标签,直接位于 data 根目录的归为'通用'"""
            rel_dir = os.path.relpath(os.path.dirname(read_path),data_path_abs)
            return rel_dir.split(os.sep)[0] if rel_dir != '.' else '通用'

        data_path_abs = get_abs_path(chroma_conf['data_path'])
        allowed_files_path = listdir_with_allowed_type(
            data_path_abs,
            tuple(chroma_conf['allow_knowledge_file_type'])
        )
        for path in allowed_files_path:
            # PDF为同名txt的派生阅读版,存在同名txt时跳过,避免重复入库
            if path.endswith('.pdf') and os.path.exists(os.path.splitext(path)[0] + '.txt'):
                logger.info(f'[加载知识库]{path}存在同名txt文件,跳过')
                continue
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f'[加载知识库]{path}内容已经存在知识库内,跳过')
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f'[加载知识库]{path}内没有有效文本内容,跳过')
                    continue
                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f'[加载知识库]{path}分片后没有有效文本内容,跳过')
                    continue
                # 为每个分片注入专科元数据,供多Agent按专科过滤检索
                department = get_department(path)
                for doc in split_document:
                    doc.metadata['department'] = department
                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f'[加载知识库]{path} 内容加载成功,专科标签: {department}')
            except Exception as e:
                logger.error(f'[加载知识库]{path} 加载失败: {str(e)}',exc_info=True)
                continue

if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    # 全库检索
    res = vs.get_retriever().invoke('高血压诊断标准')
    for r in res:
        print(r.page_content)
        print(r.metadata)
        print('*'*20)
    # 按专科过滤检索
    res = vs.get_retriever(department='心血管内科').invoke('高血压诊断标准')
    for r in res:
        print(r.page_content)
        print(r.metadata)
        print('*'*20)
