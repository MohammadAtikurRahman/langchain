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
import fs from "fs";
import csv from "csv-parser";
import stringSimilarity from "string-similarity";

const app = express();
const loader1 = new CSVLoader("./datset/all_product.csv");
const loader2 = new CSVLoader("./datset/areaCode.csv");

let product_data = [];
const deliveryFilePath = "./datset/all_product.csv";

fs.createReadStream(deliveryFilePath)
  .pipe(csv())
  .on("data", (row) => {
    // Dynamically create an object with all columns from the CSV
    let rowData = {};
    for (let key in row) {
      rowData[key] = row[key];
    }
    product_data.push(rowData);
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
      area_code: row.area_code,
      area_orginal: row.area_orginal,
      delivery_method: row.delivery_method,
      areacode_charge: row.areacode_charge,
      delivery_date: row.delivery_date,
    });
  })
  .on("end", () => {
    console.log("CSV file processing finished.");
  });

app.use(cors());
app.use(express.json());

var messageString;
async function processDocuments(loader, user_name) {
  const docs = await loader.load();
  const splitter = new CharacterTextSplitter({
    chunkSize: 10000, // or even lower depending on your data
    chunkOverlap: 7000,
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

let savedProductName = "";

function augmentMessageWithInstructions(originalMessage) {
  const naming2 = product_data; // This is an array of product data
  const productNames = naming2.map((product) => product.product_name);
  let detectedProductName = productNames.find((productName) =>
    originalMessage.includes(productName)
  );

  if (detectedProductName) {
    savedProductName = detectedProductName;
  }

  console.log("Product Name:", savedProductName);

  //Shipping Charge Case:

  if (
    originalMessage.includes("shipping") &&
    originalMessage.includes("charge")
  ) {
    return (
      originalMessage +
      " please keep the weight value in your memory and for this question don't reply, Just Display This message: What is your Areacode or Location? "
    );
  }

  const matchedArea = areaCode.find((d) => d.area_orginal === originalMessage);
  const matcharea_original = matchedArea?.area_orginal;

  const matcharea_charge = matchedArea?.areacode_charge;

  const matcharea_delivery = matchedArea?.delivery_method;
  const naming = product_data; // This is an array of product data

  const foundProduct = naming.find(
    (product) => product.product_name.trim() === savedProductName.trim()
  );

  const deliverydate = matchedArea?.delivery_date;
  // console.log("delivery date", deliverydate);

  const today = new Date();

  // Parse the delivery dates
  const deliveryDates = deliverydate?.split(", ");

  // Convert string dates into date objects
  const deliveryDateObjects = deliveryDates?.map(
    (dateStr) => new Date(dateStr)
  );

  // Sort delivery dates based on proximity to today's date
  const sortedDates = deliveryDateObjects
    ?.filter((date) => date >= today)
    .sort((a, b) => a - b);

  // Get the closest three dates
  const closestThreeDates = sortedDates?.slice(0, 3);

  // console.log("Closest three dates:", closestThreeDates);

  if (foundProduct) {
    console.log("Found Product:", foundProduct.product_name);
    console.log("Found Product:", foundProduct.weight);
    console.log("Found Product:", foundProduct.price);

    var valueofproduct = foundProduct.price;
  } else {
    console.log("Product not found.");
  }
  const transformedKey =
    "weight_based_shipping - " +
    matcharea_delivery?.toLowerCase().replace("shipping - ", "") +
    "_delivery";

  for (let product of naming) {
    if (product?.weight === foundProduct?.weight) {
      // matching by weight
      if (product.hasOwnProperty(transformedKey)) {
        var valueof = product[transformedKey];

        // console.log(`For product '${product.product_name}', with weight '${product.weight}', the matched column for '${matcharea_delivery}' is '${transformedKey}' and its value is '${product[transformedKey]}'`);
      } else {
        console.log(
          `For product '${product.product_name}', with weight '${product.weight}', no match found for '${matcharea_delivery}'`
        );
      }
    }
  }

  const chargeAsNumber = parseFloat(matcharea_charge);
  const valueAsNumber = parseFloat(valueof);
  const productpricetotall = parseFloat(valueofproduct);
  const totalCharge = chargeAsNumber + valueAsNumber;
  const totall_main_price = productpricetotall + totalCharge;

  messageString =
    "AREA CHARGE: "+matcharea_charge +" DELIVERY CHARGE: "+ valueof+ " SHIPPING CHARGE: "+ totalCharge +" PRODUCT PRICE: " + valueofproduct + " GRAND TOTAL: " + totall_main_price;



  //AREA original and AREA CODE  
  if (originalMessage.includes(matcharea_original)) {
    originalMessage = matcharea_original;

    // return originalMessage +" "+ matcharea_delivery + " " +matcharea_charge+ " " +valueof+ " " +valueofproduct

    return (
      originalMessage +
      " and this value is AREA CHARGE " +
      matcharea_charge +
      " this value is DELIVERY CHARGE " +
      valueof +
      " added this AREA CHARGE and DELIVERY CHARGE" +
      messageString +
      " and also this is show it's formate" +
      closestThreeDates +
      "this is show just it's formate"
    );
  }

  return originalMessage;
}

app.post("/api/", async (req, res) => {
  let message = req.body.message;

  const codeprice = areaCode.find((d) => d.area_code === message);

  if (codeprice?.area_orginal) {
    message = codeprice.area_orginal;
  }

  message = augmentMessageWithInstructions(message);

  console.log("prompt message", message);

  if (message) {
    const response = await handleConversation(message);

    const demoChain = new ConversationChain({
      llm: modelForDemo,
      memory: memoryForDemo,
    });
    const demoResponse = await demoChain.call({ input: message });
    //  console.log("Demo Chain Response:", demoResponse);

    const dataset_response = response;

    //  console.log("Demo Dataset Response:", dataset_response.text);

    const final_result = dataset_response.text;

    res.json({
      botResponse:
        +(messageString ? "And " + messageString : "") +
        "\n\n" +
        "System:" +
        demoResponse.response+
        "\n\n" +
        "Dataset: " + final_result

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
