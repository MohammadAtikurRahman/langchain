import * as dotenv from "dotenv";
dotenv.config();
import { LLMChain, PromptTemplate } from "langchain";


import express from "express";
import cors from "cors";
import { OpenAI } from "langchain/llms/openai";
import { BufferMemory, ChatMessageHistory , EntityMemory,
  ENTITY_MEMORY_CONVERSATION_TEMPLATE,} from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage, SystemMessage } from "langchain/schema";
import fs from "fs";
import csv from "csv-parser";

const app = express();
const loader1 = new CSVLoader("./datset/all_product.csv");
const loader2 = new CSVLoader("./datset/areaCode.csv");

let deliveryData = [];
const deliveryFilePath = "./datset/delivery.csv";

fs.createReadStream(deliveryFilePath)
  .pipe(csv())
  .on("data", (row) => {
    deliveryData.push({
      location: row.location,
      rules: row.rules,
      weight_dl: row["weight-dl"],
      operator: row.operator,
      deliveryPrice: row.deliveryPrice,
    });
  })
  .on("end", () => {
    console.log("CSV file processing finished.");
  });

let areaCode = [];
const areaCodePath = "./datset/areaCode.csv";

fs.createReadStream(areaCodePath)
  .pipe(csv())
  .on("data", (row) => {
    // Store each row in the deliveryData array as JSON
    areaCode.push({
      delivery: row.delivery,
      area_code: row.area_code,
      area_orginal: row.area_orginal,
      region: row.region,
      areacode_charge: row.areacode_charge,
    });
  })
  .on("end", () => {
    console.log("CSV file processing finished.");
  });

app.use(cors());
app.use(express.json());

async function processDocuments(loader, user_name) {
  const docs = await loader.load();
  const splitter = new CharacterTextSplitter({
    chunkSize: 3036,
    chunkOverlap: 2000,
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
  const allDocs1 = await processDocuments(loader1, "dataset");

  const allDocs2 = await processDocuments(loader2, "areacode");

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

const modelForDemo = new OpenAI({});
const chatHistoryForDemo = new ChatMessageHistory();
const memoryForDemo = new BufferMemory({ chatHistory: chatHistoryForDemo });


app.post("/api/", async (req, res) => {
  let message = req.body.message;
  if (message) {
    const response = await handleConversation(message);

    const demoChain = new ConversationChain({
      llm: modelForDemo,
      memory: memoryForDemo,
    });
    const demoResponse = await demoChain.call({ input: message });
    const dataset_response = response;
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
