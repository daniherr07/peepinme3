// --- 1. THE DEFINITIVE FIX FOR VERCEL DEPLOYMENT ---
// This forces the API route to run in the full Node.js environment,
// which is required by the @xenova/transformers library. This will solve the 405 error.
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

import { pipeline, ZeroShotClassificationPipeline, FeatureExtractionPipeline, ZeroShotClassificationOutput } from '@xenova/transformers';
import { dot, norm } from 'mathjs';
import storesData from '../../data/stores_with_embeddings.json';

// --- TYPE DEFINITIONS ---
interface ProductEmbedding { product: string; embedding: number[]; }
export interface Store { id: number; name: string; category: string; location: { province: string; city: string; }; product_embeddings: ProductEmbedding[]; product_types: string[]; hours: string; contact: string; }
interface StoreWithScore extends Store { score: number; }
export interface StoreGroup { category:string; stores: Store[]; }
export interface ChatbotResponse { introMessage: string; storeGroups?: StoreGroup[]; }

// --- In-Memory Index (For Performance) ---
// This runs only once when the server starts.
console.log("🚀 Building In-Memory Index...");
const categoryIndex = (storesData as Store[]).reduce((index, store) => {
    if (!index.has(store.category)) index.set(store.category, []);
    index.get(store.category)!.push(store);
    return index;
}, new Map<string, Store[]>());
console.log(`✅ In-Memory Index built with ${categoryIndex.size} categories.`);

// --- AI Model Management ---
class AIModels {
    private static classifier: Promise<ZeroShotClassificationPipeline> | null = null;
    private static extractor: Promise<FeatureExtractionPipeline> | null = null;
    static async getClassifier(): Promise<ZeroShotClassificationPipeline> {
        if (!this.classifier) this.classifier = pipeline('zero-shot-classification', 'Xenova/bart-large-mnli');
        return this.classifier;
    }
    static async getExtractor(): Promise<FeatureExtractionPipeline> {
        if (!this.extractor) this.extractor = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        return this.extractor;
    }
}

// --- HELPER FUNCTIONS ---
function cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    const dotProduct = Number(dot(vecA, vecB));
    const normA = Number(norm(vecA));
    const normB = Number(norm(vecB));
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (normA * normB);
}

// --- CORS HEADERS ---
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

// --- MAIN API HANDLER for POST requests ---
export async function POST(request: Request) {
    const startTime = Date.now();
    try {
        const { query } = await request.json();
        console.log(`\n--- New Request ---`);
        console.log(`[${new Date().toISOString()}] ➡️ Received query: "${query}"`);

        const classifier = await AIModels.getClassifier();
        const extractor = await AIModels.getExtractor();
        
        const candidateLabels = Array.from(categoryIndex.keys());
        const rawCategoryResults: ZeroShotClassificationOutput | ZeroShotClassificationOutput[] = await classifier(query, candidateLabels, { multi_label: true });

        if (Array.isArray(rawCategoryResults)) {
            throw new Error("Classifier returned an array for a single query, which is unexpected.");
        }
        const categoryResults: ZeroShotClassificationOutput = rawCategoryResults;
        
        const categoryScores = new Map(categoryResults.labels.map((label, i) => [label, categoryResults.scores[i]]));
        console.log(`[LOG] AI Category Scores:`, categoryScores);

        const candidateStores: Store[] = [];
        for (const [category, score] of categoryScores.entries()) {
            if (score > 0.4) candidateStores.push(...(categoryIndex.get(category) || []));
        }
        const uniqueCandidateStores = [...new Map(candidateStores.map(item => [item['id'], item])).values()];
        console.log(`[LOG] Found ${uniqueCandidateStores.length} candidate stores from relevant categories.`);
        
        if (uniqueCandidateStores.length === 0) {
            return Response.json({ introMessage: "I'm sorry, I couldn't find any stores related to that topic." }, { headers: corsHeaders });
        }

        const queryEmbeddingOutput = await extractor(query, { pooling: 'mean', normalize: true });
        const queryVector = Array.from(queryEmbeddingOutput.data);
        
        const scoredStores: StoreWithScore[] = uniqueCandidateStores.map(store => {
            let bestProductSimilarity = 0;
            for (const { embedding } of store.product_embeddings) {
                const similarity = cosineSimilarity(queryVector, embedding);
                if (similarity > bestProductSimilarity) bestProductSimilarity = similarity;
            }
            const finalScore = (categoryScores.get(store.category) || 0) * (1 + bestProductSimilarity);
            return { ...store, score: finalScore };
        });

        const topStores = scoredStores.sort((a, b) => b.score - a.score).slice(0, 5);
        console.log(`[LOG] Top 5 stores after ranking:`, topStores.map(s => ({ name: s.name, score: s.score })));

        let response: ChatbotResponse;
        if (topStores.length === 0) {
            response = { introMessage: "I found some related categories, but no specific products matched. Could you be more specific?" };
        } else {
            const groups = new Map<string, Store[]>();
            for (const store of topStores) {
                if (!groups.has(store.category)) groups.set(store.category, []);
                groups.get(store.category)!.push(store);
            }
            const finalStoreGroups = Array.from(groups.entries()).map(([category, stores]) => ({ category, stores }));
            response = {
                introMessage: "Based on your request, I found these highly relevant stores:",
                storeGroups: finalStoreGroups,
            };
        }
        
        const endTime = Date.now();
        console.log(`[LOG] ✅ Response prepared in ${endTime - startTime}ms.`);
        return Response.json(response, { headers: corsHeaders });

    } catch (error) {
        console.error("API Error:", error);
        return Response.json({ introMessage: "I'm having a little trouble thinking right now. Please try again." }, { status: 500, headers: corsHeaders });
    }
}
