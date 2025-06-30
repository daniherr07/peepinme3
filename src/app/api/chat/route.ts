// This is the new, HIGHLY OPTIMIZED server-side "brain" of the chatbot.
// It uses pre-computed indexes for near-instant lookups.

import { pipeline, ZeroShotClassificationPipeline, FeatureExtractionPipeline, ZeroShotClassificationOutput } from '@xenova/transformers';
import { dot, norm } from 'mathjs';
import storesData from '../../data/stores_with_embeddings.json';
import { NextResponse } from 'next/server';

// --- TYPE DEFINITIONS (Unchanged) ---
interface ProductEmbedding { product: string; embedding: number[]; }
export interface Store { id: number; name: string; category: string; location: { province: string; city: string; }; product_embeddings: ProductEmbedding[]; product_types: string[]; hours: string; contact: string; }
interface StoreWithScore extends Store { score: number; }
export interface StoreGroup { category: string; stores: Store[]; }
export interface ChatbotResponse { introMessage: string; storeGroups?: StoreGroup[]; }

// --- 1. PRE-COMPUTATION & IN-MEMORY INDEX ---
console.log("🚀 Building In-Memory Index...");
const categoryIndex = (storesData as Store[]).reduce((index, store) => {
    if (!index.has(store.category)) {
        index.set(store.category, []);
    }
    index.get(store.category)!.push(store);
    return index;
}, new Map<string, Store[]>());
console.log(`✅ In-Memory Index built with ${categoryIndex.size} categories.`);


// --- AI MODEL MANAGEMENT (Unchanged) ---
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

// --- HELPER FUNCTIONS (Unchanged) ---
function cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    const dotProduct = Number(dot(vecA, vecB));
    const normA = Number(norm(vecA));
    const normB = Number(norm(vecB));
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (normA * normB);
}

// --- MAIN API HANDLER (FIXED) ---
export async function POST(request: Request) {
    const startTime = Date.now();
    try {
        const { query } = await request.json();
        console.log(`\n--- New Request ---`);
        console.log(`[${new Date().toISOString()}] ➡️ Received query: "${query}"`);

        if (!query || typeof query !== 'string' || query.trim() === '') {
            return NextResponse.json({ introMessage: "Please ask me something..." }, { status: 400 });
        }

        const classifier = await AIModels.getClassifier();
        const extractor = await AIModels.getExtractor();
        
        // --- Algorithm Stage 1: AI Category Filtering ---
        const candidateLabels = Array.from(categoryIndex.keys());
        
        // --- FIX: Use a temporary variable and a type guard BEFORE strict assignment ---
        const rawCategoryResults = await classifier(query, candidateLabels, { multi_label: true });
        if (Array.isArray(rawCategoryResults)) {
            // This is an unexpected state for a single query.
            throw new Error("Classifier returned an array for a single query string.");
        }
        // Now that TypeScript knows it's not an array, we can safely assign it.
        const categoryResults: ZeroShotClassificationOutput = rawCategoryResults;
        
        const categoryScores = new Map(categoryResults.labels.map((label, i) => [label, categoryResults.scores[i]]));
        console.log(`[LOG] AI Category Scores:`, categoryScores);

        // --- Algorithm Stage 2: Index-Based Candidate Selection ---
        const candidateStores: Store[] = [];
        for (const [category, score] of categoryScores.entries()) {
            if (score > 0.4) {
                candidateStores.push(...(categoryIndex.get(category) || []));
            }
        }
        const uniqueCandidateStores = [...new Map(candidateStores.map(item => [item['id'], item])).values()];
        console.log(`[LOG] Found ${uniqueCandidateStores.length} candidate stores from relevant categories.`);
        
        if (uniqueCandidateStores.length === 0) {
            console.log(`[LOG] No relevant categories found. Sending fallback.`);
            const endTime = Date.now();
            console.log(`[LOG] ✅ Response prepared in ${endTime - startTime}ms.`);
            return NextResponse.json({ introMessage: "I'm sorry, I couldn't find any stores related to that topic." });
        }

        // --- Algorithm Stage 3: Semantic Ranking on the SMALL SUBSET ---
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

        // --- Final Filtering, Sorting, and Grouping ---
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
        console.log(`---------------------`);
        return NextResponse.json(response);

    } catch (error) {
        console.error("API Error:", error);
        return NextResponse.json({ introMessage: "I'm having a little trouble thinking right now. Please try again." }, { status: 500 });
    }
}