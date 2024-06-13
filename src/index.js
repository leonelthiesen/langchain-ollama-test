// https://js.langchain.com/v0.1/docs/get_started/quickstart/

import { ChatOllama } from "@langchain/community/chat_models/ollama";

const chatModel = new ChatOllama({
    baseUrl: "http://localhost:11434", // Default value
    model: "mistral",
});

await chatModel.invoke("what is LangSmith?");


import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a world class technical documentation writer."],
    ["user", "{input}"],
]);

const chain = prompt.pipe(chatModel);

await chain.invoke({
    input: "what is LangSmith?",
});

import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

const llmChain = prompt.pipe(chatModel).pipe(outputParser);

await llmChain.invoke({
    input: "what is LangSmith?",
});
