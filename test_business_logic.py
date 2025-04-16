import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Import the modules you want to test.
# Adjust the import paths if your project structure differs.
from langchain.schema.document import Document

from app import database
from app import get_embedding_function
from app import rag


# --- Helper functions to create dummy Documents ---
def create_dummy_document(text, source="dummy.pdf", page=1):
    metadata = {"source": source, "page": page}
    return Document(page_content=text, metadata=metadata)


class TestDatabaseFunctions(unittest.TestCase):

    def test_calculate_chunk_ids(self):
        # Create two dummy documents representing two chunks from the same page.
        dummy_chunks = [
            create_dummy_document("Chunk one", source="file1.pdf", page=2),
            create_dummy_document("Chunk two", source="file1.pdf", page=2),
            create_dummy_document("Chunk three", source="file1.pdf", page=3),
        ]
        # Run calculate_chunk_ids on these dummy chunks.
        updated_chunks = database.calculate_chunk_ids(dummy_chunks)
        
        # The first two chunks (file1.pdf:2) should have same current_id so second gets an index of 1.
        self.assertEqual(updated_chunks[0].metadata["id"], "file1.pdf:2:0")
        self.assertEqual(updated_chunks[1].metadata["id"], "file1.pdf:2:1")
        # The third chunk should reset the count.
        self.assertEqual(updated_chunks[2].metadata["id"], "file1.pdf:3:0")

    def test_split_documents(self):
        # Create a dummy document with content longer than chunk_size
        long_text = "A" * 2000  # a simple repeated character, 2000 chars total
        dummy_doc = create_dummy_document(long_text)
        chunks = database.split_documents([dummy_doc])
        
        # Using your configuration, chunk_size=800 and overlap=80
        # We expect multiple chunks that overlap.
        self.assertTrue(len(chunks) > 1)
        # Check that each chunk is at most 800 chars (except perhaps the last one).
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 800)

    def test_clear_database(self):
        # Create a temporary directory to simulate CHROMA_PATH.
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create a dummy file inside the temp directory.
            dummy_file = os.path.join(tmp_dir, "dummy.txt")
            with open(dummy_file, "w") as f:
                f.write("test")
            # Patch CHROMA_PATH in the database module to point to tmp_dir.
            with patch.object(database, "CHROMA_PATH", tmp_dir):
                # Ensure the file exists.
                self.assertTrue(os.path.exists(dummy_file))
                # Call clear_database and then verify the folder is gone.
                database.clear_database()
                self.assertFalse(os.path.exists(tmp_dir))
        finally:
            # Cleanup in case deletion failed.
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

    def test_get_embedding_function(self):
        # Call the embedding function.
        embedding_fn = get_embedding_function.get_embedding_function()
        # Check that the returned object has the attribute 'model_name' matching our config.
        self.assertEqual(embedding_fn.model_name, "sentence-transformers/all-MiniLM-L6-v2")


class TestRAGFunction(unittest.TestCase):
    def test_query_rag(self):
        # Create dummy documents to be returned by the vectorstore.
        dummy_doc = create_dummy_document("Context information", source="test.pdf", page=1)
        dummy_result = [(dummy_doc, 0.1)]

        # Patch the Chroma class in rag.py to return a dummy instance.
        with patch("app.rag.Chroma") as DummyChroma:
            dummy_db_instance = MagicMock()
            # Set up similarity_search_with_score to return our dummy result.
            dummy_db_instance.similarity_search_with_score.return_value = dummy_result
            DummyChroma.return_value = dummy_db_instance

            # Patch the Ollama model so that it returns a dummy answer.
            with patch("rag.Ollama") as DummyOllama:
                dummy_llm_instance = MagicMock()
                dummy_llm_instance.invoke.return_value = "Dummy answer from LLM."
                DummyOllama.return_value = dummy_llm_instance

                # Now call the query_rag function with a dummy question.
                result = rag.query_rag("What is the context?")
                # Check if the answer and sources are as expected.
                self.assertEqual(result["response"], "Dummy answer from LLM.")
                self.assertTrue("sources" in result)
                self.assertEqual(result["sources"][0]["source"], "test.pdf")


if __name__ == "__main__":
    unittest.main()
