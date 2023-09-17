import * as dotenv from "dotenv";
dotenv.config();

import { OpenAI } from "langchain/llms/openai";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { CSVLoader } from "langchain/document_loaders/fs/csv";

async function loadDocuments() {
    const loader = new CSVLoader("./dataset.csv");
    return await loader.load();
}

async function processDocuments(docs, user_name) {
    const splitter = new CharacterTextSplitter({
        chunkSize: 1536,
        chunkOverlap: 200,
    });

    const allDocs = [];
    for (const doc of docs) {
        const pageContent = doc.pageContent;
        const chunkHeader = `DOCUMENT NAME: ${user_name}\n\n---\n\n`;
        const createdDocs = await splitter.createDocuments([pageContent], [], {
            chunkHeader,
            appendChunkOverlapHeader: true,
        });
        allDocs.push(...createdDocs);
    }

    return allDocs;
}

async function getVectorStore(documents) {
    return await HNSWLib.fromDocuments(documents, new OpenAIEmbeddings());
}

async function handleConversation(input, qaChain, conversationChain) {
    const resFromQAChain = await qaChain.call({ query: input });
    
    if (resFromQAChain.text && resFromQAChain.text.trim() !== '') {
        return resFromQAChain.text;
    }

    const resFromConversationChain = await conversationChain.call({ input: input });
    if (resFromConversationChain.response && resFromConversationChain.response.text && resFromConversationChain.response.text.trim() !== '') {
        return resFromConversationChain.response.text;
    }

    return undefined;
}

async function main() {
    const docs = await loadDocuments();
    const processedDocs = await processDocuments(docs, "1:8 Pagani Huayra Bc Bricks Assemble Car");
    const vectorStore = await getVectorStore(processedDocs);

    const qaModel = new OpenAI({ temperature: 0 });
    const qaChain = new RetrievalQAChain({
        combineDocumentsChain: loadQAStuffChain(qaModel),
        retriever: vectorStore.asRetriever(),
        returnSourceDocuments: true,
    });

    const conversationModel = new OpenAI({});
    const memory = new BufferMemory();
    const conversationChain = new ConversationChain({
        llm: conversationModel,
        memory: memory,
    });

    console.log(await handleConversation("What is the sku of 1:8 Pagani Huayra Bc Bricks Assemble Car?", qaChain, conversationChain));
    console.log(await handleConversation("wha is price of 1:8 Pagani Huayra Bc Bricks Assemble Car", qaChain, conversationChain));
}

main();
