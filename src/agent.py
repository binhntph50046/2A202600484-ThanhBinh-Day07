from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Retrieval-augmented generation (RAG):
        1. Retrieve: Get relevant chunks.
        2. Augment: Build prompt with context.
        3. Generate: Call LLM.
        """
        # Step 1: Retrieve
        search_results = self.store.search(question, top_k=top_k)
        
        if not search_results:
            return "Tôi không tìm thấy thông tin trong tài liệu"
            
        # Step 2: Augment
        context_text = "\n\n".join([res["content"] for res in search_results])
        
        prompt = (
            "Bạn là một trợ lý ảo chuyên nghiệp. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp sau đây.\n"
            "Nếu ngữ cảnh không chứa thông tin cần thiết, hãy trả lời chính xác là: \"Tôi không tìm thấy thông tin trong tài liệu\".\n"
            "Không được bịa đặt thông tin hoặc sử dụng kiến thức bên ngoài ngữ cảnh này.\n\n"
            f"Ngữ cảnh:\n{context_text}\n\n"
            f"Câu hỏi: {question}\n"
            "Trả lời:"
        )
        
        # Step 3: Generate
        return self.llm_fn(prompt)
