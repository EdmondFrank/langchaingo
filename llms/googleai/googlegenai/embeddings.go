package googlegenai

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// CreateEmbedding creates embeddings from texts.
func (g *GoogleAI) CreateEmbedding(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, 0, len(texts))

	// Process texts in batches of 100
	for i := 0; i < len(texts); i += 100 {
		end := i + 100
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]

		// Convert texts to Content objects
		contents := make([]*genai.Content, len(batch))
		for j, text := range batch {
			contents[j] = &genai.Content{
				Parts: []*genai.Part{{Text: text}},
			}
		}

		// Call EmbedContent with the batch
		response, err := g.client.Models.EmbedContent(ctx, g.opts.DefaultEmbeddingModel, contents, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create embeddings: %w", err)
		}

		// Extract embeddings from response
		for _, embedding := range response.Embeddings {
			results = append(results, embedding.Values)
		}
	}

	return results, nil
}
