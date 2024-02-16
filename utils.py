from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import git
import os
import redis
from queue import Queue
from langchain.vectorstores.redis import Redis

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
allowed_extensions_default = ['.py', '.ipynb', '.md']

from langchain.chains import ConversationalRetrievalChain

class Embedder:
    def __init__(self, git_link) -> None:
        self.git_link = git_link
        last_name = self.git_link.split('/')[-1]
        self.clone_path = last_name.split('.')[0]
        self.llm = self.load_llm()
        self.hf = HuggingFaceEmbeddings(model_name=model_name)
        self.MyQueue =  Queue(maxsize=2)
        self.redis_conn = redis.Redis()
        self.redis_url="redis://localhost:6379"

    def add_to_queue(self, value):
        if self.MyQueue.full():
            self.MyQueue.get()
        self.MyQueue.put(value)

    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            # Clone the repository
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_files(self, allowed_extensions=allowed_extensions_default):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try:
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e:
                        pass

    def extract_docs_files(self):
        self.extract_files(allowed_extensions=['.md'])

    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)
        self.num_texts = len(self.texts)

    def embed_deeplake(self):
        # db = DeepLake(dataset_path=self.deeplake_path, embedding_function= OpenAIEmbeddings())
        db = DeepLake(dataset_path=self.deeplake_path, embedding_function= self.hf)
        db.add_documents(self.texts)
        ## Remove data from the cloned path
        self.delete_directory(self.clone_path)
        return db

    def index_exists(self, idx_name="docs_idx"):
        res = self.redis_conn.ft(idx_name).info()
        print(res)

    def redis_store(self):
        db = Redis.from_documents(self.texts, self.hf, index_name="idx", redis_url=self.redis_url)
        ## Remove data from the cloned path
        self.delete_directory(self.clone_path)

        return db

    def delete_directory(self, path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)

    def load_db(self):
        exists = self.index_exists()
        if exists:
            ## Just load the DB
            print("redis idx already exixtss")

            # self.db = DeepLake(
            # dataset_path=self.deeplake_path,
            # read_only=True,
            # embedding_function=self.hf,
            #  )
        else:
            ## Create and load
            self.extract_docs_files()
            self.chunk_files()
            self.db = self.redis_store()

        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def load_llm():
        """load the llm"""

        llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q2_K.bin', # model available here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                        model_type='llama',
                        config={'max_new_tokens': 256, 'temperature': 0})
        return llm

    def create_prompt_template():
        # prepare the template that provides instructions to the chatbot

        template = """Use the provided context to answer the user's question.
        If you don't know the answer, respond with "I do not know".
        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question'])
        return prompt

    def retrieve_results(self, query):
        qa = RetrievalQA.from_chain_type(llm=self.llm,
                                        chain_type='stuff',
                                        retriever=self.retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': self.create_prompt_template()})
        # qa = ConversationalRetrievalChain.from_llm(self.model, chain_type="stuff", retriever=self.retriever, condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'))
       # result = qa({"question": query, "chat_history": chat_history})
        result = qa({"question": query})
        self.add_to_queue((query, result["answer"]))
        return result['answer']
