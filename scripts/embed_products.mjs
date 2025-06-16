// This script runs ONCE to add AI-powered semantic embeddings to your store data.
// It reads your stores.json, uses an AI model to analyze each product type,
// and saves a new file with this "semantic knowledge" baked in.

import { pipeline } from '@xenova/transformers';
import fs from 'fs';
import path from 'path';

// Define the path to your data files
const STORES_INPUT_PATH = path.resolve(process.cwd(), 'src/app/data/stores.json');
const STORES_OUTPUT_PATH = path.resolve(process.cwd(), 'src/app/data/stores_with_embeddings.json');

// The AI model we'll use for converting text to semantic vectors (embeddings)
// This is a small, fast, and powerful model perfect for this task.
const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';

async function generateEmbeddings() {
  console.log('Starting the embedding process. This may take a few minutes...');

  // 1. Load the AI feature-extraction model
  const extractor = await pipeline('feature-extraction', MODEL_NAME);
  console.log('AI model loaded.');

  // 2. Load your original store data
  const storesRaw = fs.readFileSync(STORES_INPUT_PATH);
  const stores = JSON.parse(storesRaw);
  console.log(`Loaded ${stores.length} stores.`);

  // 3. Process each store and its products
  const storesWithEmbeddings = [];
  for (const store of stores) {
    console.log(`Processing store: ${store.name}`);
    
    const productEmbeddings = [];
    for (const product of store.product_types) {
      // Generate the vector embedding for the product text
      const output = await extractor(product, { pooling: 'mean', normalize: true });
      // Convert the model's output to a standard array
      const embedding = Array.from(output.data);
      
      productEmbeddings.push({
        product: product,
        embedding: embedding,
      });
    }

    storesWithEmbeddings.push({
      ...store,
      // Replace the old product_types with our new, smarter structure
      product_embeddings: productEmbeddings,
    });
  }

  // 4. Save the new, enhanced data to a file
  fs.writeFileSync(STORES_OUTPUT_PATH, JSON.stringify(storesWithEmbeddings, null, 2));
  console.log(`\nSuccessfully created ${STORES_OUTPUT_PATH}`);
  console.log('You can now run your Next.js app!');
}

generateEmbeddings();