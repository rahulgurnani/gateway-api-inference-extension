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

package dataproducer

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrmm "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/multimodalencoder"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

const (
	MultimodalEncoderPluginType = "multimodal-encoder-data-producer"
)

// MultimodalEncoderPlugin is a plugin interface for encoding multimodal data in LLM requests.
type MultimodalEncoderPlugin struct {
}

// TypedName returns the type and name of the plugin.
func (p *MultimodalEncoderPlugin) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: MultimodalEncoderPluginType,
		Name: MultimodalEncoderPluginType,
	}
}

func (p *MultimodalEncoderPlugin) Produces() map[string]any {
	return map[string]any{attrmm.MultimodalDataKey: attrmm.MultimodalData{}}
}

func (p *MultimodalEncoderPlugin) Consumes() map[string]any {
	return map[string]any{}
}

// PrepareRequestData is the main hook for the MultimodalEncoderPlugin.
func (p *MultimodalEncoderPlugin) PrepareRequestData(ctx context.Context, request *schedulingtypes.LLMRequest, endpoints []schedulingtypes.Endpoint) error {
	logger := log.FromContext(ctx)
	if request.Body == nil {
		logger.V(logutil.VERBOSE).Info("PrepareRequestData: request body is nil, skipping")
		return nil
	}

	var items []attrmm.MultimodalItem

	// 1. Handle ChatCompletions
	if request.Body.ChatCompletions != nil {
		logger.V(logutil.VERBOSE).Info("PrepareRequestData: scanning ChatCompletions messages", "messageCount", len(request.Body.ChatCompletions.Messages))
		for _, msg := range request.Body.ChatCompletions.Messages {
			for _, block := range msg.Content.Structured {
				if item := extractFromBlock(ctx, block); item != nil {
					items = append(items, *item)
				}
			}
		}
	}

	// 2. Handle Responses (/v1/responses)
	// OpenAI Realtime/Responses API: input is an array of items.
	// Each item can have "content" which is an array of content parts.
	if request.Body.Responses != nil {
		if inputSlice, ok := request.Body.Responses.Input.([]any); ok {
			logger.V(logutil.VERBOSE).Info("PrepareRequestData: scanning Responses input", "itemCount", len(inputSlice))
			for _, item := range inputSlice {
				if itemMap, ok := item.(map[string]any); ok {
					if content, ok := itemMap["content"]; ok {
						if contentSlice, ok := content.([]any); ok {
							for _, part := range contentSlice {
								if partMap, ok := part.(map[string]any); ok {
									if mmItem := extractFromMap(ctx, partMap); mmItem != nil {
										items = append(items, *mmItem)
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// 3. Handle Conversations (/v1/conversations)
	if request.Body.Conversations != nil {
		logger.V(logutil.VERBOSE).Info("PrepareRequestData: scanning Conversations items", "itemCount", len(request.Body.Conversations.Items))
		for _, convItem := range request.Body.Conversations.Items {
			// Content can be a string or an array of content parts
			if contentSlice, ok := convItem.Content.([]any); ok {
				for _, part := range contentSlice {
					if partMap, ok := part.(map[string]any); ok {
						if mmItem := extractFromMap(ctx, partMap); mmItem != nil {
							items = append(items, *mmItem)
						}
					}
				}
			} else if contentMap, ok := convItem.Content.(map[string]any); ok {
				// Single content part
				if mmItem := extractFromMap(ctx, contentMap); mmItem != nil {
					items = append(items, *mmItem)
				}
			}
		}
	}

	if len(items) == 0 {
		logger.V(logutil.VERBOSE).Info("PrepareRequestData: no multimodal items found")
		return nil
	}

	logger.V(logutil.VERBOSE).Info("PrepareRequestData: storing multimodal data", "itemCount", len(items), "endpointCount", len(endpoints))
	mmData := &attrmm.MultimodalData{Items: items}

	for _, endpoint := range endpoints {
		endpoint.Put(attrmm.MultimodalDataKey, mmData)
	}

	return nil
}

func extractFromBlock(ctx context.Context, block schedulingtypes.ContentBlock) *attrmm.MultimodalItem {
	logger := log.FromContext(ctx)
	logger.V(logutil.DEBUG).Info("extractFromBlock: processing block", "type", block.Type)
	var data []byte
	switch block.Type {
	case "image_url":
		if strings.HasPrefix(block.ImageURL.Url, "data:") {
			commaIdx := strings.Index(block.ImageURL.Url, ",")
			if commaIdx != -1 {
				dataStr := block.ImageURL.Url[commaIdx+1:]
				if strings.Contains(block.ImageURL.Url[:commaIdx], "base64") {
					if d, err := base64.StdEncoding.DecodeString(dataStr); err == nil {
						data = d
					} else {
						logger.V(logutil.DEBUG).Info("extractFromBlock: failed to decode image_url base64", "err", err)
					}
				}
			}
		} else {
			logger.V(logutil.DEBUG).Info("extractFromBlock: image_url is an external URL, skipping")
		}
	case "input_audio":
		if d, err := base64.StdEncoding.DecodeString(block.InputAudio.Data); err == nil {
			data = d
		} else {
			logger.V(logutil.DEBUG).Info("extractFromBlock: failed to decode input_audio base64", "err", err)
		}
	}

	if len(data) > 0 {
		hash := sha256.Sum256(data)
		item := &attrmm.MultimodalItem{
			Data: data,
			Hash: hex.EncodeToString(hash[:]),
		}
		logger.V(logutil.DEBUG).Info("extractFromBlock: extracted item", "type", block.Type, "sizeBytes", len(data), "hash", item.Hash)
		return item
	}
	return nil
}

func extractFromMap(ctx context.Context, m map[string]any) *attrmm.MultimodalItem {
	logger := log.FromContext(ctx)
	t, _ := m["type"].(string)
	logger.V(logutil.DEBUG).Info("extractFromMap: processing part", "type", t)
	var data []byte
	switch t {
	case "image_url":
		iu, _ := m["image_url"].(map[string]any)
		url, _ := iu["url"].(string)
		if strings.HasPrefix(url, "data:") {
			commaIdx := strings.Index(url, ",")
			if commaIdx != -1 {
				dataStr := url[commaIdx+1:]
				if strings.Contains(url[:commaIdx], "base64") {
					if d, err := base64.StdEncoding.DecodeString(dataStr); err == nil {
						data = d
					} else {
						logger.V(logutil.DEBUG).Info("extractFromMap: failed to decode image_url base64", "err", err)
					}
				}
			}
		} else {
			logger.V(logutil.DEBUG).Info("extractFromMap: image_url is an external URL, skipping")
		}
	case "input_audio":
		ia, _ := m["input_audio"].(map[string]any)
		dataStr, _ := ia["data"].(string)
		if d, err := base64.StdEncoding.DecodeString(dataStr); err == nil {
			data = d
		} else {
			logger.V(logutil.DEBUG).Info("extractFromMap: failed to decode input_audio base64", "err", err)
		}
	}

	if len(data) > 0 {
		hash := sha256.Sum256(data)
		item := &attrmm.MultimodalItem{
			Data: data,
			Hash: hex.EncodeToString(hash[:]),
		}
		logger.V(logutil.DEBUG).Info("extractFromMap: extracted item", "type", t, "sizeBytes", len(data), "hash", item.Hash)
		return item
	}
	return nil
}
