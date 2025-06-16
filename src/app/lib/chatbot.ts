/**
 * PeepInMe Chatbot Logic - V8.1 (Corrected Types)
 *
 * This version introduces major performance and correctness improvements:
 * 1. High-Speed Performance: The AI classifier is now called only ONCE per query.
 * 2. Type-Safe Math: Correctly handles numeric types from the mathjs library.
 * 3. Two-stage AI filtering remains for high accuracy.
 */

import { pipeline, ZeroShotClassificationPipeline, FeatureExtractionPipeline, ZeroShotClassificationOutput } from '@xenova/transformers';
import { dot, norm } from 'mathjs'; // Import math functions
import storesData from '../data/stores_with_embeddings.json'; // Use the data with pre-calculated embeddings

// --- TYPE DEFINITIONS (No changes) ---
interface ProductEmbedding {
  product: string;
  embedding: number[];
}
export interface Store {
  id: number;
  name: string;
  category: string;
  location: { province: string; city: string; };
  product_embeddings: ProductEmbedding[];
  product_types: string[];
  hours: string;
  contact: string;
}
interface StoreWithScore extends Store {
  score: number;
}
export interface StoreGroup {
    category: string;
    stores: Store[];
}
export interface ChatbotResponse {
  introMessage: string;
  storeGroups?: StoreGroup[];
}

// --- AI MODEL MANAGEMENT (No changes) ---
class AIModels {
  private static classifier: Promise<ZeroShotClassificationPipeline> | null = null;
  private static extractor: Promise<FeatureExtractionPipeline> | null = null;
  static async getClassifier(): Promise<ZeroShotClassificationPipeline> {
    if (this.classifier === null) this.classifier = pipeline('zero-shot-classification', 'Xenova/bart-large-mnli') as Promise<ZeroShotClassificationPipeline>;
    return this.classifier;
  }
  static async getExtractor(): Promise<FeatureExtractionPipeline> {
    if (this.extractor === null) this.extractor = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2') as Promise<FeatureExtractionPipeline>;
    return this.extractor;
  }
}

// --- HELPER FUNCTIONS (No changes) ---
const getRandomMessage = (messages: string[]) => messages[Math.floor(Math.random() * messages.length)];
function cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    const dotProduct = Number(dot(vecA, vecB));
    const normA = Number(norm(vecA));
    const normB = Number(norm(vecB));
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (normA * normB);
}

// --- MAIN QUERY PROCESSING FUNCTION (FIXED) ---
export async function processQuery(query: string): Promise<ChatbotResponse> {
  if (!query || query.trim() === '') return { introMessage: "Please ask me something..." };

  try {
    const classifier = await AIModels.getClassifier();
    const extractor = await AIModels.getExtractor();
    const candidateLabels = [...new Set(storesData.map(store => store.category))];

    // --- AI Stage 1: Fast Category Classification ---
    const rawCategoryResults = await classifier(query, candidateLabels, { multi_label: true });

    // --- FIX: Add a type guard to handle the array possibility ---
    if (Array.isArray(rawCategoryResults)) {
      // This is an unexpected state for a single query. We'll treat it as a failure.
      throw new Error("Classifier returned an array for a single query string.");
    }
    const categoryResults: ZeroShotClassificationOutput = rawCategoryResults;
    // After this check, TypeScript knows `categoryResults` is a single object.

    // Create a score map for O(1) lookups
    const categoryScores = new Map(categoryResults.labels.map((label, i) => [label, categoryResults.scores[i]]));

    // --- AI Stage 2: Deep Semantic Analysis of User Query ---
    const queryEmbeddingOutput = await extractor(query, { pooling: 'mean', normalize: true });
    const queryVector = Array.from(queryEmbeddingOutput.data);

    // --- Scoring Logic (No changes) ---
    const scoredStores: StoreWithScore[] = storesData.map(store => {
      const categoryScore = categoryScores.get(store.category) || 0;
      let bestProductSimilarity = 0;
      for (const { embedding } of store.product_embeddings) {
          const similarity = cosineSimilarity(queryVector, embedding);
          if (similarity > bestProductSimilarity) {
              bestProductSimilarity = similarity;
          }
      }
      const finalScore = categoryScore * (1 + bestProductSimilarity);
      return { ...store, score: finalScore };
    });

    // --- Filtering, Sorting, and Grouping (No changes) ---
    const topStores = scoredStores
      .filter(store => store.score > 0.5)
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    if (topStores.length === 0) {
      return { introMessage: "I'm sorry, I couldn't find any stores that specifically match your request. Could you try rephrasing?" };
    }

    const groups = new Map<string, Store[]>();
    for (const store of topStores) {
        if (!groups.has(store.category)) groups.set(store.category, []);
        groups.get(store.category)!.push(store);
    }

    const finalStoreGroups = Array.from(groups.entries()).map(([category, stores]) => ({ category, stores }));

    return {
      introMessage: "Based on your request, I found these highly relevant stores:",
      storeGroups: finalStoreGroups,
    };
  } catch (error) {
    console.error("Error processing query:", error);
    return { introMessage: "I'm having a little trouble thinking right now. Please try again." };
  }
}