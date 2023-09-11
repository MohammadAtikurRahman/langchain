import * as dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import { OpenAI } from "langchain/llms/openai";
import { BufferMemory, ChatMessageHistory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage, SystemMessage } from "langchain/schema";
// Import the required modules using ES6 module syntax
import fs from "fs";
import csv from "csv-parser";

// Your code here

const app = express();

const loader1 = new CSVLoader("./datset/dataset.csv");
const loader2 = new CSVLoader("./datset/areaCode.csv");
// ... Add more loaders for additional CSVs ...

app.use(cors());
app.use(express.json());

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

let qaChain, conversationChain;

async function setupChains() {
  const allDocs1 = await processDocuments(loader1, "User Name 1");
  const allDocs2 = await processDocuments(loader2, "User Name 2");
  const allDocs = [...allDocs1, ...allDocs2];

  const vectorStore = await HNSWLib.fromDocuments(
    allDocs,
    new OpenAIEmbeddings()
  );

  const qaModel = new OpenAI({ temperature: 0 });
  qaChain = new RetrievalQAChain({
    combineDocumentsChain: loadQAStuffChain(qaModel),
    retriever: vectorStore.asRetriever(),
    returnSourceDocuments: true,
  });

  const conversationModel = new OpenAI({});
  const memory = new BufferMemory();
  conversationChain = new ConversationChain({
    llm: conversationModel,
    memory: memory,
  });
}

async function handleConversation(input) {
  const resFromQAChain = await qaChain.call({ query: input });

  if (resFromQAChain.text && resFromQAChain.text.trim() !== "") {
    chatHistoryForDemo.addUserMessage(input); // Save user input to chat history
    chatHistoryForDemo.addAIChatMessage(resFromQAChain.text); // Save AI response to chat history
    return resFromQAChain;
  }

  const resFromConversationChain = await conversationChain.call({
    input: input,
  });

  if (
    resFromConversationChain.response &&
    resFromConversationChain.response.text &&
    resFromConversationChain.response.text.trim() !== ""
  ) {
    chatHistoryForDemo.addUserMessage(input); // Save user input to chat history
    chatHistoryForDemo.addAIChatMessage(resFromConversationChain.response.text); // Save AI response to chat history
    return resFromConversationChain;
  }

  return { text: "Unable to find an answer." };
}

//const chat = new ChatOpenAI();

// Move these outside the endpoint to maintain a single, persistent instance across requests.
const modelForDemo = new OpenAI({});
const chatHistoryForDemo = new ChatMessageHistory();
const memoryForDemo = new BufferMemory({ chatHistory: chatHistoryForDemo });

app.post("/api/", async (req, res) => {
  let message = req.body.message;

  let promptPrice = "and";
  if (message.includes("price")) {
    message = message + " " + promptPrice;
    console.log("prompt", message);
  }

  if (message) {
    const response = await handleConversation(message);

    // Use the persistent demoChain for conversation context.
    const demoChain = new ConversationChain({
      llm: modelForDemo,
      memory: memoryForDemo,
    });
    const demoResponse = await demoChain.call({ input: message });
    console.log("Demo Chain Response:", demoResponse);

    const dataset_response = response;

    console.log("Demo Dataset Response:", dataset_response.text);

    const final_result = dataset_response.text;

    res.json({
      botResponse:
        "\n\n" +
        "Dataset:" +
        final_result +
        "\n\n" +
        "System:" +
        demoResponse.response,
    });
    return;
  }

  res.status(400).send({ error: "Message is required!" });
});

const PORT = 5000;
setupChains()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
    });
  })
  .catch((error) => {
    console.error("Failed to setup chains:", error);
  });
