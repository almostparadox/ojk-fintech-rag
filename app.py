import json
import streamlit as st
from src.config import settings, LegalChunk
from src.retrieval.hybrid_search import HybridSearcher
from src.generation.client import LegalGenerator
from src.ingestion.indexer import LegalIndexer

st.set_page_config(
    page_title="OJK Fintech Regulatory RAG",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ OJK & Fintech Legal Intelligence Assistant")
st.markdown("Asisten Kepatuhan Hukum Fintech Indonesia berbasis **Hybrid RAG** (LanceDB + BM25) dengan validasi status keberlakuan peraturan.")

with st.sidebar:
    st.header("⚙️ Pengaturan & Filter")
    active_only = st.checkbox("Hanya Peraturan Aktif (Berlaku)", value=True, help="Saring otomatis aturan yang telah dicabut (misal POJK 77/2016)")
    top_k = st.slider("Jumlah Rujukan Pasal (Top K)", min_value=1, max_value=8, value=3)

    st.divider()
    st.subheader("📚 Status Basis Data")
    if st.button("🔄 Reload / Re-index Sample Data"):
        with st.spinner("Mengindeks data sample..."):
            all_chunks = []
            for f in (settings.DATA_DIR / "sample").glob("*.json"):
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    all_chunks.extend([LegalChunk(**item) for item in data])
            LegalIndexer().index_chunks(all_chunks)
            st.success("Basis data berhasil diperbarui!")

    st.markdown("""
    ---
    **Peraturan Terindeks:**
    - `POJK 10/POJK.05/2022` (P2P Lending)
    - `SEOJK 19/SEOJK.06/2023` (Bunga & Etika Penagihan)
    - `UU 27/2022` (Pelindungan Data Pribadi)
    - `POJK 77/POJK.01/2016` (Status: Dicabut)
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "contexts" in msg:
            with st.expander("🔍 Lihat Dasar Hukum yang Dirujuk"):
                for c in msg["contexts"]:
                    st.markdown(f"**{c['legal_ref']}** `[Status: {c['status']}]`")
                    st.text(c['content'])

if prompt := st.chat_input("Tanyakan aturan hukum (contoh: Berapa modal disetor fintech lending?):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            searcher = HybridSearcher()
            results = searcher.search(prompt, top_k=top_k, active_only=active_only)
        except FileNotFoundError:
            warning_msg = (
                "⚠️ Basis data regulasi belum diindeks! "
                "Silakan klik tombol **'🔄 Reload / Re-index Sample Data'** di sidebar kiri, "
                "atau jalankan `python cli.py bootstrap` di terminal terlebih dahulu."
            )
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
            st.stop()

        if not results:
            ans = "Berdasarkan peraturan yang tersedia dalam basis data, dasar hukum untuk pertanyaan ini tidak ditemukan."
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        else:
            with st.expander("🔍 Dasar Hukum Ditemukan (RRF Hybrid Search)", expanded=False):
                for r in results:
                    status_badge = "🟢 BERLAKU" if r.chunk.status == "Berlaku" else "🔴 DICABUT"
                    st.markdown(f"**{r.chunk.legal_ref}** — {status_badge} (Score: `{r.score:.4f}`)")
                    st.caption(f"{r.chunk.bab}")
                    st.code(r.chunk.content, language="text")

            generator = LegalGenerator()
            contexts = [r.chunk for r in results]
            response_container = st.empty()
            full_response = ""

            try:
                for chunk in generator.stream_response(prompt, contexts):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
            except Exception as e:
                full_response = f"⚠️ Gagal menghubungi LLM: {e}\n\nPastikan `NINEROUTER_API_KEY` terkonfigurasi di `.env`."
                response_container.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "contexts": [r.chunk.model_dump() for r in results]
            })
