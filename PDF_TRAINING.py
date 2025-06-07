!pip install torch transformers sentence-transformers chromadb PyPDF2 nltk

from google.colab import files
uploaded = files.upload()

import torch
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    pipeline, T5Tokenizer, T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb, PyPDF2, re
import numpy as np
from typing import List, Dict
import warnings, nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

warnings.filterwarnings('ignore')

class AdvancedTransformerPDFQA:
    def __init__(self):


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self._initialize_models()

        self.client = chromadb.Client()
        self.collection = None

        self.chunks = []
        self.chunk_size = 400
        self.overlap = 150
        self.max_context_length = 2048

        print("System initialized successfully!")

    def _initialize_models(self):

        print("Loading Q&A model (BERT)...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
        self.qa_model.to(self.device)

        print("Loading text generation model (T5)...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
        self.t5_model.to(self.device)

        print("Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

        print("Loading cross-encoder for re-ranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print("Loading advanced Q&A pipeline...")
        self.advanced_qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )

        print("All transformer models loaded successfully!")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        print(f"Extracting text from {pdf_path}...")

        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                print(f"Total pages: {total_pages}")

                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_page = self._clean_page_text(page_text)
                            text += cleaned_page + "\n\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
                        continue

                    if page_num % 50 == 0:
                        print(f"Processed {page_num + 1}/{total_pages} pages...")

                print(f"Extracted {len(text)} characters from {total_pages} pages")
                return text

        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""

    def _clean_page_text(self, text: str) -> str:
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        return text.strip()

    def advanced_text_cleaning(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-"\'\n]', '', text)
        sentences = sent_tokenize(text)
        clean_sentences = []
        for sentence in sentences:
            if len(sentence.split()) >= 3 and len(sentence) >= 10:
                clean_sentences.append(sentence.strip())
        return ' '.join(clean_sentences)

    def intelligent_chunking(self, text: str) -> List[Dict]:
        print("Creating intelligent text chunks...")

        if not text:
            return []

        sentences = sent_tokenize(text)
        print(f"Found {len(sentences)} sentences")

        chunks = []
        current_chunk = ""
        current_sentences = []

        for i, sentence in enumerate(sentences):
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk.split()) > self.chunk_size and current_chunk:
                chunk_data = {
                    'text': current_chunk.strip(),
                    'sentences': current_sentences.copy(),
                    'start_sentence': i - len(current_sentences),
                    'end_sentence': i - 1,
                    'embedding': None
                }
                chunks.append(chunk_data)
                overlap_sentences = current_sentences[-self.overlap//50:] if current_sentences else []
                current_chunk = " ".join(overlap_sentences + [sentence])
                current_sentences = overlap_sentences + [sentence]
            else:
                current_chunk = test_chunk
                current_sentences.append(sentence)

        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'sentences': current_sentences,
                'start_sentence': len(sentences) - len(current_sentences),
                'end_sentence': len(sentences) - 1,
                'embedding': None
            }
            chunks.append(chunk_data)

        chunks = [chunk for chunk in chunks if len(chunk['text'].split()) >= 20]

        print(f"Created {len(chunks)} intelligent chunks")
        return chunks

    def create_advanced_embeddings(self, chunks: List[Dict]):
        print("Creating advanced embeddings...")

        try:
            try:
                self.client.delete_collection("advanced_pdf_collection")
            except:
                pass

            self.collection = self.client.create_collection(
                name="advanced_pdf_collection",
                metadata={"hnsw:space": "cosine"}
            )

            batch_size = 64
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                texts = [chunk['text'] for chunk in batch_chunks]
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=True,
                    batch_size=16
                ).tolist()
                for j, embedding in enumerate(embeddings):
                    batch_chunks[j]['embedding'] = embedding
                ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
                metadatas = [
                    {
                        'chunk_index': i+j,
                        'start_sentence': chunk['start_sentence'],
                        'end_sentence': chunk['end_sentence'],
                        'sentence_count': len(chunk['sentences']),
                        'text_length': len(chunk['text'])
                    }
                    for j, chunk in enumerate(batch_chunks)
                ]
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

            self.chunks = chunks
            print("Advanced embeddings created and stored!")
            return True

        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
            return False

    def classify_question_type(self, question: str) -> Dict:
        question_lower = question.lower().strip()
        question_types = {
            'factual': ['what', 'who', 'when', 'where', 'which'],
            'explanatory': ['how', 'why', 'explain', 'describe'],
            'yes_no': ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would'],
            'comparative': ['compare', 'difference', 'better', 'versus', 'vs'],
            'procedural': ['steps', 'process', 'procedure', 'method'],
            'definitional': ['define', 'definition', 'meaning', 'means']
        }
        detected_type = 'general'
        confidence = 0.5
        for q_type, keywords in question_types.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_type = q_type
                confidence = 0.8
                break
        return {
            'type': detected_type,
            'confidence': confidence,
            'question': question
        }

    def retrieve_relevant_contexts(self, question: str, n_results: int = 10) -> List[Dict]:
        if not self.collection:
            return []
        try:
            question_embedding = self.embedding_model.encode([question]).tolist()
            results = self.collection.query(
                query_embeddings=question_embedding,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            contexts = []
            if results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    contexts.append({
                        'text': doc,
                        'metadata': metadata,
                        'similarity': 1 - distance,
                        'rank': i + 1
                    })
            # Cross-encoder re-ranking
            pairs = [(question, ctx['text']) for ctx in contexts]
            scores = self.cross_encoder.predict(pairs)
            for ctx, score in zip(contexts, scores):
                ctx['cross_score'] = score
            contexts = sorted(contexts, key=lambda x: x['cross_score'], reverse=True)
            return contexts
        except Exception as e:
            print(f"Error retrieving contexts: {e}")
            return []

    def extract_relevant_sentences(self, question: str, contexts: List[Dict], max_sentences: int = 5) -> str:
        question_lower = question.lower()
        question_words = set(question_lower.split())
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'of', 'to', 'for', 'and', 'or', 'but'}
        question_words = question_words - stop_words
        relevant_sentences = []
        for ctx in contexts:
            sentences = sent_tokenize(ctx['text'])
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for word in question_words if word in sentence_lower)
                if score > 0:
                    relevant_sentences.append((sentence, score))
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in relevant_sentences[:max_sentences]]
        return ' '.join(top_sentences)

    def generate_comprehensive_answer(self, question: str, contexts: List[Dict]) -> str:
        if not contexts:
            return "I couldn't find relevant information to answer your question."

        question_info = self.classify_question_type(question)
        question_lower = question.lower()
        is_definition = any(word in question_lower for word in ['what does', 'what is', 'define', 'definition', 'meaning', 'mean'])

    # Combine top N context chunks for summarization
        combined_context = ' '.join([ctx['text'] for ctx in contexts[:3]])
        if len(combined_context) > 1500:
            combined_context = combined_context[:1500]

        try:
        # Always use T5 for summarization/answering
            if is_definition:
                input_text = f"Based on the following text, explain what the term means: {question} Text: {combined_context}"
            elif question_info['type'] == 'explanatory':
                input_text = f"Based on the following text, provide a detailed explanation: {question} Text: {combined_context}"
            else:
                input_text = f"Summarize the following to answer the question: {question} Text: {combined_context}"

            inputs = self.t5_tokenizer(
                input_text,
                return_tensors='pt',
                max_length=1024,
                truncation=True,
                padding=True
        )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=250,
                    min_length=30,
                    num_beams=4,
                    temperature=0.2,
                    do_sample=True,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
            )

            answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If T5 output is too short or unclear, fallback to extractive QA
            if not answer or len(answer.split()) < 15:
                try:
                    qa_result = self.advanced_qa_pipeline(
                        question=question,
                        context=combined_context
                )
                    if qa_result['score'] > 0.1 and qa_result['answer']:
                      return qa_result['answer']
                except Exception as e:
                    print(f"Fallback QA pipeline error: {e}")

            if answer:
              answer = answer.strip()
            # Ensure answer ends with a period
              if not answer.endswith(('.', '!', '?')):
                  answer += '.'
              return answer

            return "I couldn't find a clear answer to your question in the document."

        except Exception as e:
              print(f"Error in comprehensive generation: {e}")
              return "Error"


    def answer_question(self, question: str) -> str:
        if not self.collection:
            return "System not trained. Please upload a PDF first."
        contexts = self.retrieve_relevant_contexts(question, n_results=10)
        if not contexts or (contexts[0]['cross_score'] < 0.3):  # You can tune this threshold
            return "Sorry, I couldn't find relevant information in the document."
        answer = self.generate_comprehensive_answer(question, contexts)
        return answer

    def train_on_pdf(self, pdf_path: str) -> bool:
        print("="*50)
        print("STARTING ADVANCED TRANSFORMER TRAINING")
        print("="*50)
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            print("Failed to extract text")
            return False
        clean_text = self.advanced_text_cleaning(raw_text)
        if not clean_text:
            print("Text cleaning failed")
            return False
        print(f"Processed text: {len(clean_text)} characters")
        chunks = self.intelligent_chunking(clean_text)
        if not chunks:
            print("Failed to create chunks")
            return False
        success = self.create_advanced_embeddings(chunks)
        if not success:
            print("Failed to create embeddings")
            return False
        print("="*50)
        print("ADVANCED TRAINING COMPLETED!")
        print("="*50)
        print(f"Total chunks: {len(chunks)}")
        return True

    def main():
        print("Initializing Advanced Transformer PDF Q&A System")
        qa_system = AdvancedTransformerPDFQA()
        pdf_path = "CompaniesAct2013.pdf"
        print(f"\nTraining on: {pdf_path}")
        if qa_system.train_on_pdf(pdf_path):
            print("\nTraining successful! Ready for questions.")
            return qa_system
        else:
            print("Training failed!")
            return None

    def interactive_session(qa_system):
        if not qa_system:
            print("System not initialized!")
            return
        print("\n" + "="*40)
        print("Q&A SESSION")
        print("="*40)
        print("Type 'quit' to exit")
        print("-"*40)
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("Thank you!")
                break
            if question.lower() == 'help':
                print("Ask specific questions about document content")
                continue
            if not question:
                continue
            answer = qa_system.answer_question(question)
            print(f"\n{answer}")

if __name__ == "__main__":
    qa_system = main()
    if qa_system:
        interactive_session(qa_system)
