import * as dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import { OpenAI } from "langchain/llms/openai";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage, SystemMessage } from "langchain/schema";

const app = express();

const loader1 = new CSVLoader("./dataset.csv");
const loader2 = new CSVLoader("./areaCode.csv");
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
    return resFromQAChain.text;
  }

  const resFromConversationChain = await conversationChain.call({
    input: input,
  });

  if (
    resFromConversationChain.response &&
    resFromConversationChain.response.text &&
    resFromConversationChain.response.text.trim() !== ""
  ) {
    return resFromConversationChain.response.text;
  }

  return "Unable to find an answer.";
}
const chat = new ChatOpenAI();

app.post("/api/", async (req, res) => {
  const message = req.body.message;

  if (message) {
    const response = await handleConversation(message);

    if (message.includes("price")) {
      console.log("1st resposne:", response);

      const response2 = await chat.call([
        new SystemMessage(
          "You are a helpful assistant that just ask what is your delivery location"
        ),
        new HumanMessage("Say grettings and ask what is your locataion? "),
      ]);

      console.log(response2);

      res.json({ botResponse: "\n\n" + response + "\n\n" + response2.content });
      return;
    } else {
      console.log("2nd resposne:", response);

      res.json({ botResponse: "\n\n" + response });
      return;
    }
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
