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
import fs from 'fs';
import csv from 'csv-parser';

// Your code here


const app = express();

const loader1 = new CSVLoader("./datset/dataset.csv");
const loader2 = new CSVLoader("./datset/delivery.csv");
const loader3 = new CSVLoader("./datset/areaCode.csv");


// ... Add more loaders for additional CSVs ...

let deliveryData = []; // To store the data as a global JSON variable

const deliveryFilePath = "./datset/delivery.csv";

fs.createReadStream(deliveryFilePath)
  .pipe(csv())
  .on('data', (row) => {
    // Store each row in the deliveryData array as JSON
    deliveryData.push({
      location: row.location,
      rules: row.rules,
      weight_dl: row['weight-dl'],
      operator: row.operator,
      deliveryPrice: row.deliveryPrice,
    });
  })
  .on('end', () => {
    console.log('CSV file processing finished.');
  });



  let areaCode = [];
  const areaCodePath = "./datset/areaCode.csv"


fs.createReadStream(areaCodePath)
  .pipe(csv())
  .on('data', (row) => {
    // Store each row in the deliveryData array as JSON
    areaCode.push({
      delivery: row.delivery,
      area_code: row.area_code,
      area_orginal: row.area_orginal,
      region: row.region,
      areacode_charge: row.areacode_charge,
    });
  })
  .on('end', () => {
    console.log('CSV file processing finished.');
  });









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

  const allDocs3 = await processDocuments(loader3, "User Name 3");

  const allDocs = [...allDocs1, ...allDocs2, ...allDocs3];

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
    chatHistoryForDemo.addUserMessage(input);  // Save user input to chat history
    chatHistoryForDemo.addAIChatMessage(resFromQAChain.text);  // Save AI response to chat history
    return resFromQAChain;
  }

  const resFromConversationChain = await conversationChain.call({ input: input });

  if (
    resFromConversationChain.response &&
    resFromConversationChain.response.text &&
    resFromConversationChain.response.text.trim() !== ""
  ) {
    chatHistoryForDemo.addUserMessage(input);  // Save user input to chat history
    chatHistoryForDemo.addAIChatMessage(resFromConversationChain.response.text);  // Save AI response to chat history
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
   


  let promptPrice = "and please answer it with it's weight very shortly";
  if (message.includes("price")) {
    message = message + " " + promptPrice;
    console.log("prompt", message);
  } 



   
  const locationExists = deliveryData.some((item) => message.includes(item.location));
   let promptDelivery = " this location for rules,weight-dl,operator,deliveryPrice and location and previous weight-dl based delivery charge and please dont show others all comparsion";

  if (locationExists) {

      message = message + " " + promptDelivery;

    // console.log("Message contains a valid location from deliveryData.");
    // console.log(locationExists);
    // console.log("prompt", message)
  } 



  let promptArea = " it has areacode_charge and areacode_charge can be 0 or 39 and it show please and this is 0 or 39 and please tell one area_orginal wise area_charge and this charge please added previous totall price"
  const matchedArea = areaCode.find(d => d.area_code === message);

  if (matchedArea) {
      console.log(matchedArea.area_orginal);
      const area_with_charge = matchedArea.area_orginal;


      message = area_with_charge + " " + promptArea;

      console.log(message)


  }
  




  // let promptDelivery = "and if you get region,code,area then this wise you will get a charge and charge can 39,79 (39+79) or 0,79 (0+79) or 39,0 (39+0) or 0,0 (0+0) of addition added previous final price";
  // if(message.includes("code")) {
  //   message = message + " " + promptDelivery;
  //   console.log("prompt", message);
  // } 
   
  






  if (message) {
    const response = await handleConversation(message);

    // Use the persistent demoChain for conversation context.
    const demoChain = new ConversationChain({ llm: modelForDemo, memory: memoryForDemo });
    const demoResponse = await demoChain.call({ input: message });
    console.log("Demo Chain Response:", demoResponse);

    const dataset_response = response;


    console.log("Demo Dataset Response:", dataset_response.text);

    const final_result= dataset_response.text;

    res.json({ botResponse: "\n\n" + "Dataset:" +final_result + "\n\n" + "System:" +demoResponse.response });
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
