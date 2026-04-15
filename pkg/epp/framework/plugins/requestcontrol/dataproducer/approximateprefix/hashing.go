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
	"github.com/zeebo/blake3"
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

func hashMultimodalBlock(block requesthandling.ContentBlock) (string, error) {
	b, err := json.Marshal(block)
	if err != nil {
		return "", err
	}
	hash := blake3.Sum256(b)
	return hex.EncodeToString(hash[:]), nil
}

func isMultimodalUrl(block requesthandling.ContentBlock) bool {
	if block.Type == "image_url" || block.Type == "video_url" {
		// the image_url field is slightly a misnomer—it doesn't just take web links. It natively supports Base64 encoded data using the Data URI scheme.
		if strings.HasPrefix(block.ImageURL.Url, "data:image") || strings.HasPrefix(block.VideoURL.Url, "data:video") {
			return false
		}
		return true
	} 
	return false
}

// hashMultimodalContent hashes the multimodal content of a message if it contains multimodal content.
func hashMultimodalContentIfPresent(msg requesthandling.Message) ([]any, error) {
	var combined []any
	// explicitly adding role to the hash to differentiate between messages with same content but different roles.
	combined = append(combined, msg.Role)
	for _, block := range msg.Content.Structured {
		if isMultimodalUrl(block) {
			// Always hash multimodal URLs to avoid long URLs being part of the hash, which cause hotspotting.
			hash, err := hashMultimodalBlock(block)
			if err != nil {
				return nil, err
			}
			combined = append(combined, hash)
		} else if block.Type == "text" {
			combined = append(combined, block.Text)
		} else {
			b, err := json.Marshal(block)
			if err != nil {
				return nil, err
			}
			combined = append(combined, string(b))
		} 
	}
	return combined, nil
}

func parseChatCompletions(messages []requesthandling.Message) ([]any, error) {
	var res []any
	for _, msg := range messages {
		if msg.Content.Raw != "" {
			res = append(res, msg.Content.Raw)
		} else if len(msg.Content.Structured) > 0 {
			hashedContent, err := hashMultimodalContentIfPresent(msg)
			if err != nil {
				return nil, err
			}
			res = append(res, hashedContent)
		}
	}
	return res, nil
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
