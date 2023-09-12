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
const loader2 = new CSVLoader("./datset/areaCode.csv");


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

  const allDocs = [...allDocs1,...allDocs2];

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


const modelForDemo = new OpenAI({});
const chatHistoryForDemo = new ChatMessageHistory();
const memoryForDemo = new BufferMemory({ chatHistory: chatHistoryForDemo });



function augmentMessageWithInstructions(originalMessage) {
  
  if (originalMessage.includes("price")&& !originalMessage.includes("previous")) {
    return originalMessage + "and this product name also Give a short answer with just the weight";
  }

  // if (/\d+/.test(originalMessage)) {
  //   const matchedArea = areaCode.find(d => d.area_code === originalMessage);
  //   originalMessage = matchedArea;
  //   console.log("inside the areacode",originalMessage.area_orginal)
  //   return originalMessage.area_orginal + "";
    
  // }


  // const matchedArea = areaCode.find(d => d.area_orginal === originalMessage);
  // const matcharea_original = matchedArea.area_orginal;
  // if (originalMessage.includes(matcharea_original)) {

  //   originalMessage = matcharea_original;
  //   return originalMessage + " is area_orginal. area_original has areacode_charge row wise .. this areacode_charge added with previous product price";
  // }



  
  return originalMessage; // Return the original message if no conditions matched
}

function addCustomTextToResponse(datasetResponse, demoResponse) {
  let customText = "";

  if (datasetResponse.includes("price")) {
    customText += "What is your location ";
  }

  if (demoResponse.includes("price")) {
    customText += "or Area Code or Area ?";
  }

  return customText;
}



app.post("/api/", async (req, res) => {
  let message = req.body.message;

  message = augmentMessageWithInstructions(message);

  console.log("prompt message",message);

  if (message) {
    const response = await handleConversation(message);

    const demoChain = new ConversationChain({ llm: modelForDemo, memory: memoryForDemo });
    const demoResponse = await demoChain.call({ input: message });
  //  console.log("Demo Chain Response:", demoResponse);

    const dataset_response = response;


  //  console.log("Demo Dataset Response:", dataset_response.text);

    const final_result= dataset_response.text;


    const customText = addCustomTextToResponse(final_result, demoResponse.response);


    res.json({ botResponse: "\n\n" + "Dataset:" +final_result + "\n\n" + "System:" +demoResponse.response + "\n\n"+ customText });
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
