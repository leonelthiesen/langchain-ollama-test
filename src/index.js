// https://js.langchain.com/v0.1/docs/get_started/quickstart/

import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    maxConcurrency: 5,
});

const loader = new CheerioWebBaseLoader("https://docs.smith.langchain.com/user_guide");
const splitter = new RecursiveCharacterTextSplitter();

const docs = await loader.load();
const splitDocs = await splitter.splitDocuments(docs);
const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

const chatModel = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: "tinyllama",
});

const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt,
});

const retriever = vectorstore.asRetriever();

const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
});

const outputParser = new StringOutputParser();

console.log(await retrievalChain.invoke({
    input: "what is LangSmith?",
}));
