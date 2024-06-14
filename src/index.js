// https://js.langchain.com/v0.1/docs/get_started/quickstart/

import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

import 'dotenv/config';


const embeddings = new OpenAIEmbeddings();

const loader = new CheerioWebBaseLoader("https://docs.smith.langchain.com/user_guide");
const splitter = new RecursiveCharacterTextSplitter();

const docs = await loader.load();
const splitDocs = await splitter.splitDocuments(docs);
const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

const chatModel = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY
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
