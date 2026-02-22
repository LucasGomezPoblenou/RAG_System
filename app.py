import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

st.set_page_config(page_title="Asistente II Guerra Mundial")
st.title("Chatbot Segunda guerra Mundial")
st.markdown("---")

query = st.text_input("Introduce tu pregunta:")

if query:
    with st.spinner("Buscando en los documentos..."):

        docs = retriever.invoke(query)

        if docs:
            contexto = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
            Eres un asistente histórico experto en la Segunda Guerra Mundial.

            Responde a la pregunta del usuario utilizando SOLO la información proporcionada en el CONTEXTO.
            Sé muy detallado. Si te preguntan por países, causas o conferencias, intenta listar todo lo que aparezca.

            Si la información no está en el contexto, di: "No encuentro información sobre esto en los documentos proporcionados."

            CONTEXTO:
            {contexto}

            PREGUNTA:
            {query}

            RESPUESTA:
            """

            response = llm.invoke(prompt)

            st.subheader("Respuesta del Asistente:")
            st.write(response.content)

            with st.expander("Ver evidencias de los documentos (Fuentes)"):
                for i, doc in enumerate(docs):
                    st.info(f"Fragmento {i + 1}:\n\n{doc.page_content[:400]}...")
        else:
            st.warning("No se han encontrado fragmentos relevantes en la base de datos.")