import * as dotenv from "dotenv";
dotenv.config();
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";

import { CSVLoader } from "langchain/document_loaders/fs/csv";

const loader = new CSVLoader("./office_dataset.csv");
// ... your existing code ...


const docs = await loader.load();

// console.log(docs);

const splitter = new CharacterTextSplitter({
  chunkSize: 1536,
  chunkOverlap: 200,
});

const allDocs = [];
for (const doc of docs) {
  const pageContent = doc.pageContent;
  const user_name = "Md Niaj Shahriar Shishir";
  const chunkHeader = `DOCUMENT NAME: ${user_name}\n\n---\n\n`;
  const createdDocs = await splitter.createDocuments(
    [pageContent],
    [],
    {
      chunkHeader,
      appendChunkOverlapHeader: true,
    }
  );
  allDocs.push(...createdDocs);
}

const vectorStore = await HNSWLib.fromDocuments(allDocs, new OpenAIEmbeddings());

const model = new OpenAI({ temperature: 0 });

const chain = new RetrievalQAChain({
  combineDocumentsChain: loadQAStuffChain(model),
  retriever: vectorStore.asRetriever(),
  returnSourceDocuments: true,
});




const queryuser_name = "Md Shahriar Shishir";
const queryField = "phone"; // or "sku", "brand", etc.
const query = `What is the ${queryField} of ${queryuser_name}?`;

const res = await chain.call({ query });
//console.log(JSON.stringify(res, null, 2));

console.log(res.text);
