/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package approximateprefix

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// hashPrompt divides the prompt into blocks and calculates a prefix cache hash for each block.
// The first block hash includes the model name and cache salt (if provided).
// For subsequent blocks, the hash is calculated as: hash(block i content, hash(i-1)).
func hashPrompt(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) []blockHash {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil
	}

	userInput, err := getUserInputBytes(request)
	if err != nil {
		loggerDebug.Error(err, "Failed to get user input bytes")
		return nil
	}

	// convert block size from tokens to characters
	cacheBlockSizeChars := blockSizeTokens * averageCharactersPerToken

	if cacheBlockSizeChars <= 0 {
		loggerDebug.Info("Skipping prefix hashing: block size in characters must be positive",
			"blockSizeTokens", blockSizeTokens,
			"cacheBlockSizeChars", cacheBlockSizeChars)
		return nil
	}

	if len(userInput) < cacheBlockSizeChars {
		loggerDebug.Info("Request body too small for prefix cache", "size", len(userInput), "block size in chars", cacheBlockSizeChars)
		return nil
	}

	if len(userInput) > cacheBlockSizeChars*maxPrefixBlocks {
		loggerDebug.Info("Truncating input", "size", len(userInput), "max prefix blocks", maxPrefixBlocks, "block size in chars", cacheBlockSizeChars)
		userInput = userInput[:maxPrefixBlocks*cacheBlockSizeChars]
	}

	// Split the body into blocks of size cacheBlockSizeChars.
	res := make([]blockHash, 0, len(userInput)/cacheBlockSizeChars)

	h := xxhash.New()
	// Different models should have different hashes even with the same body.
	_, _ = h.Write([]byte(request.TargetModel))
	if cacheSalt := request.Body.CacheSalt(); cacheSalt != "" {
		_, _ = h.Write([]byte(cacheSalt))
	}

	prevBlockHash := blockHash(h.Sum64())
	for i := 0; i+cacheBlockSizeChars <= len(userInput); i += cacheBlockSizeChars {
		h.Reset()
		_, _ = h.Write(userInput[i : i+cacheBlockSizeChars])
		_, _ = h.Write(toBytes(prevBlockHash))
		res = append(res, blockHash(h.Sum64()))

		prevBlockHash = res[len(res)-1]
	}

	return res
}

func toBytes(i blockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
}

// hashMultimodalContent returns a 128-character hash of the multimodal data by repeating a 64-bit xxhash 8 times.
func hashMultimodalContent(block requesthandling.ContentBlock) (string, error) {
	var dataToHash string
	switch block.Type {
	case "image_url":
		dataToHash = block.ImageURL.Url
	case "video_url":
		dataToHash = block.VideoURL.Url
	case "input_audio":
		dataToHash = block.InputAudio.Data
	default:
		b, err := json.Marshal(block)
		if err != nil {
			return "", err
		}
		dataToHash = string(b)
	}

	data := []byte(dataToHash)
	h := xxhash.Sum64(data)
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, h)
	hashStr := hex.EncodeToString(out)
	// max block size vLLM supports is 32 tokens. Each token is on average 4 chars. 4*32=128 characters.
	// hashStr is 16 characters. 8 * 16 = 128 characters.
	return strings.Repeat(hashStr, 8), nil
}

func isMultimodalContentType(contentType string) bool {
	return contentType == "image_url" || contentType == "video_url" || contentType == "input_audio"
}

// replaceMultimodalURLsInBlocks returns a copy of blocks with multimodal URL/data values replaced
// by fixed-length hashes, so that long URLs with shared prefixes don't collide in prefix cache keys.
func replaceMultimodalURLsInBlocks(blocks []requesthandling.ContentBlock) ([]requesthandling.ContentBlock, error) {
	result := make([]requesthandling.ContentBlock, len(blocks))
	for i, block := range blocks {
		if !isMultimodalContentType(block.Type) {
			result[i] = block
			continue
		}
		hash, err := hashMultimodalContent(block)
		if err != nil {
			return nil, err
		}
		result[i] = requesthandling.ContentBlock{Type: block.Type}
		switch block.Type {
		case "image_url":
			result[i].ImageURL = requesthandling.ImageBlock{Url: hash}
		case "video_url":
			result[i].VideoURL = requesthandling.VideoBlock{Url: hash}
		case "input_audio":
			result[i].InputAudio = requesthandling.AudioBlock{Data: hash, Format: block.InputAudio.Format}
		}
	}
	return result, nil
}

// parseChatCompletions returns a copy of the messages with multimodal URLs replaced by their hashes.
// The full Message struct (including role and content structure) is preserved and marshaled as-is,
// making the resulting bytes easy to reason about.
func parseChatCompletions(messages []requesthandling.Message) ([]requesthandling.Message, error) {
	result := make([]requesthandling.Message, len(messages))
	for i, msg := range messages {
		result[i] = requesthandling.Message{Role: msg.Role}
		if msg.Content.Raw != "" {
			result[i].Content = requesthandling.Content{Raw: msg.Content.Raw}
			continue
		}
		blocks, err := replaceMultimodalURLsInBlocks(msg.Content.Structured)
		if err != nil {
			return nil, err
		}
		result[i].Content = requesthandling.Content{Structured: blocks}
	}
	return result, nil
}

func getUserInputBytes(request *scheduling.InferenceRequest) ([]byte, error) {
	switch {
	case request.Body.Conversations != nil:
		return json.Marshal(request.Body.Conversations.Items)

	case request.Body.Responses != nil:
		// TODO(#2172): Parse multimodal content in responses API as well and hash multimodal URLs.
		var combined []map[string]interface{}
		if request.Body.Responses.Instructions != nil {
			combined = append(combined, map[string]interface{}{"instructions": request.Body.Responses.Instructions})
		}
		if request.Body.Responses.Tools != nil {
			combined = append(combined, map[string]interface{}{"tools": request.Body.Responses.Tools})
		}
		combined = append(combined, map[string]interface{}{"input": request.Body.Responses.Input})
		return json.Marshal(combined)

	case request.Body.ChatCompletions != nil:
		res, err := parseChatCompletions(request.Body.ChatCompletions.Messages)
		if err != nil {
			return nil, err
		}
		return json.Marshal(res)

	case request.Body.Completions != nil:
		return []byte(request.Body.Completions.Prompt.PlainText()), nil

	case request.Body.Embeddings != nil:
		// Handle embeddings API - marshal input for cache key generation
		return json.Marshal(request.Body.Embeddings.Input)

	default:
		return nil, errors.New("invalid request body: no recognized API format found")
	}
}
