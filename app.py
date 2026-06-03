"""
Day 19-20 — RAG 项目 UI（Streamlit）
智能文档问答系统
"""
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
import tempfile

# ===== 加载环境变量 =====
load_dotenv()

# ===== 配置常量 =====
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "mimo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/bge-large-zh-v1.5")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")

# ===== 页面配置 =====
st.set_page_config(
    page_title="智能文档问答系统",
    page_icon="📚",
    layout="wide"
)

# ===== 侧边栏配置 =====
with st.sidebar:
    st.title("⚙️ 配置")

    # LLM 配置
    api_key = st.text_input("API Key", type="password", value=API_KEY)
    base_url = st.text_input("API 端点", value=BASE_URL)
    model_name = st.text_input("模型名", value=MODEL_NAME)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    k_value = st.slider("检索数量 k", 1, 10, 3)

    st.divider()
    st.markdown("### 📊 系统状态")

# ===== 主界面 =====
st.title("📚 智能文档问答系统")
st.markdown("基于 RAG 的私有文档问答，支持 PDF/Markdown/Word/TXT")

# ===== 文件上传 =====
uploaded_files = st.file_uploader(
    "上传文档",
    type=["pdf", "md", "txt", "docx"],
    accept_multiple_files=True,
    help="支持 PDF、Markdown、TXT、Word 文档"
)

# ===== 初始化 session state =====
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# ===== 处理上传的文件 =====
if uploaded_files and api_key:
    with st.spinner("正在处理文档..."):
        # 加载文档
        all_docs = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            ext = os.path.splitext(file.name)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext in [".md", ".txt"]:
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name
            all_docs.extend(docs)
            os.unlink(tmp_path)

        # 切片
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ",", " ", "，", "。", "！", "？"]
        )
        chunks = splitter.split_documents(all_docs)

        # Embedding + 存储
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        persist_dir = PERSIST_DIR
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)

        vectorstore = Chroma.from_documents(
            chunks, embedding, persist_directory=persist_dir
        )
        st.session_state.vectorstore = vectorstore

        # 配置 LLM
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        # Prompt
        prompt = PromptTemplate(
            template="""你是一个专业的文档问答助手。请根据以下参考内容和对话历史回答问题。

参考内容：
{context}

对话历史：
{chat_history}

问题：{question}

要求：
1. 只根据参考内容回答，不要编造
2. 如果参考内容中没有相关信息，请回答"根据提供的文档，我没有找到相关信息"
3. 回答要准确、简洁

回答：""",
            input_variables=["context", "chat_history", "question"]
        )

        # 对话链
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": k_value}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        st.session_state.conversation = conversation

        # 更新状态
        st.session_state.uploaded = True
        st.sidebar.success(f"✅ 已加载 {len(chunks)} 个片段")
        st.sidebar.info(f"📄 {len(all_docs)} 页文档")

# ===== 对话界面 =====
if st.session_state.conversation:
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📎 查看来源"):
                    for source in msg["sources"]:
                        st.markdown(f"**{source['page']}**: {source['content'][:200]}...")

    # 用户输入
    if question := st.chat_input("输入你的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = st.session_state.conversation.invoke({"question": question})
                answer = result["answer"]
                sources = [
                    {
                        "page": doc.metadata.get("source", "未知"),
                        "content": doc.page_content
                    }
                    for doc in result.get("source_documents", [])
                ]

            st.markdown(answer)
            if sources:
                with st.expander("📎 查看来源"):
                    for source in sources:
                        st.markdown(f"**{source['page']}**: {source['content'][:200]}...")

        # 保存到历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
else:
    st.info("👆 请先上传文档并配置 API Key")

# ===== 底部信息 =====
st.divider()
st.markdown("""
<small>技术栈：LangChain + ChromaDB + bge-large-zh-v1.5 + Streamlit | 
基于 RAG 的私有文档问答系统</small>
""", unsafe_allow_html=True)
