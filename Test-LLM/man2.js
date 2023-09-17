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

const loader1 = new CSVLoader("./dataset.csv");
const loader2 = new CSVLoader("./delivery.csv");
const loader3 = new CSVLoader("./areaCode.csv");  // New loader for third CSV
const loader4 = new CSVLoader("./office_dataset.csv");  // New loader for fourth CSV

async function processDocuments(loader, user_name) {
    const docs = await loader.load();
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

async function main() {
    const allDocs1 = await processDocuments(loader1, "User Name 1");
    const allDocs2 = await processDocuments(loader2, "User Name 2");
    const allDocs3 = await processDocuments(loader3, "User Name 3"); // Process third CSV
    const allDocs4 = await processDocuments(loader4, "User Name 4"); // Process fourth CSV

    const allDocs = [...allDocs1, ...allDocs2, ...allDocs3, ...allDocs4];

    const vectorStore = await HNSWLib.fromDocuments(allDocs, new OpenAIEmbeddings());

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

    const response1 = await handleConversation("What is the sku of 1:8 Pagani Huayra Bc Bricks Assemble Car?", qaChain, conversationChain);
    console.log(response1);

    const response2 = await handleConversation("what is Shipping - Bay of Plenty deliveryPrice in weight-dl 79 ", qaChain, conversationChain);
    console.log(response2);
    
    // Queries for the new CSVs can be added here:
    const response3 = await handleConversation("3177 codes charge", qaChain, conversationChain);
    console.log(response3);
    
    const response4 = await handleConversation("previous all question you anylisis and tell me the summery", qaChain, conversationChain);
    console.log(response4);
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

    return "Unable to find an answer.";
}

main().catch(error => {
    console.error(error);
});
