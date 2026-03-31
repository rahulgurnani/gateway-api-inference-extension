/*
Copyright 2025 The Kubernetes Authors.

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
	"reflect"
	"testing"

	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrmm "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/multimodalencoder"
)

func TestMultimodalEncoderPlugin_PrepareRequestData(t *testing.T) {
	plugin := &MultimodalEncoderPlugin{}
	ctx := context.Background()

	imageData := []byte("fake-image-data")
	imageBase64 := base64.StdEncoding.EncodeToString(imageData)
	imageURL := "data:image/png;base64," + imageBase64
	imageHash := sha256.Sum256(imageData)
	imageHashStr := hex.EncodeToString(imageHash[:])

	audioData := []byte("fake-audio-data")
	audioBase64 := base64.StdEncoding.EncodeToString(audioData)
	audioHash := sha256.Sum256(audioData)
	audioHashStr := hex.EncodeToString(audioHash[:])

	tests := []struct {
		name          string
		request       *schedulingtypes.LLMRequest
		wantItems     []attrmm.MultimodalItem
		wantEndpoints bool // whether we expect data to be put in endpoints
	}{
		{
			name:    "nil body",
			request: &schedulingtypes.LLMRequest{Body: nil},
		},
		{
			name: "chat completions with image and audio",
			request: &schedulingtypes.LLMRequest{
				Body: &schedulingtypes.LLMRequestBody{
					ChatCompletions: &schedulingtypes.ChatCompletionsRequest{
						Messages: []schedulingtypes.Message{
							{
								Content: schedulingtypes.Content{
									Structured: []schedulingtypes.ContentBlock{
										{
											Type: "image_url",
											ImageURL: schedulingtypes.ImageBlock{
												Url: imageURL,
											},
										},
										{
											Type: "input_audio",
											InputAudio: schedulingtypes.AudioBlock{
												Data: audioBase64,
											},
										},
										{
											Type: "text",
											Text: "hello",
										},
									},
								},
							},
						},
					},
				},
			},
			wantItems: []attrmm.MultimodalItem{
				{Data: imageData, Size: len(imageData), Hash: imageHashStr},
				{Data: audioData, Size: len(audioData), Hash: audioHashStr},
			},
			wantEndpoints: true,
		},
		{
			name: "responses API with multimodal data",
			request: &schedulingtypes.LLMRequest{
				Body: &schedulingtypes.LLMRequestBody{
					Responses: &schedulingtypes.ResponsesRequest{
						Input: []any{
							map[string]any{
								"type": "message",
								"content": []any{
									map[string]any{
										"type": "image_url",
										"image_url": map[string]any{
											"url": imageURL,
										},
									},
								},
							},
						},
					},
				},
			},
			wantItems: []attrmm.MultimodalItem{
				{Data: imageData, Size: len(imageData), Hash: imageHashStr},
			},
			wantEndpoints: true,
		},
		{
			name: "conversations API with multimodal data - array content",
			request: &schedulingtypes.LLMRequest{
				Body: &schedulingtypes.LLMRequestBody{
					Conversations: &schedulingtypes.ConversationsRequest{
						Items: []schedulingtypes.ConversationItem{
							{
								Type: "message",
								Content: []any{
									map[string]any{
										"type": "input_audio",
										"input_audio": map[string]any{
											"data": audioBase64,
										},
									},
								},
							},
						},
					},
				},
			},
			wantItems: []attrmm.MultimodalItem{
				{Data: audioData, Size: len(audioData), Hash: audioHashStr},
			},
			wantEndpoints: true,
		},
		{
			name: "conversations API with multimodal data - single content",
			request: &schedulingtypes.LLMRequest{
				Body: &schedulingtypes.LLMRequestBody{
					Conversations: &schedulingtypes.ConversationsRequest{
						Items: []schedulingtypes.ConversationItem{
							{
								Type: "message",
								Content: map[string]any{
									"type": "image_url",
									"image_url": map[string]any{
										"url": imageURL,
									},
								},
							},
						},
					},
				},
			},
			wantItems: []attrmm.MultimodalItem{
				{Data: imageData, Size: len(imageData), Hash: imageHashStr},
			},
			wantEndpoints: true,
		},
		{
			name: "no multimodal data",
			request: &schedulingtypes.LLMRequest{
				Body: &schedulingtypes.LLMRequestBody{
					ChatCompletions: &schedulingtypes.ChatCompletionsRequest{
						Messages: []schedulingtypes.Message{
							{
								Content: schedulingtypes.Content{
									Raw: "just text",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "invalid base64 image",
			request: &schedulingtypes.LLMRequest{
				Body: &schedulingtypes.LLMRequestBody{
					ChatCompletions: &schedulingtypes.ChatCompletionsRequest{
						Messages: []schedulingtypes.Message{
							{
								Content: schedulingtypes.Content{
									Structured: []schedulingtypes.ContentBlock{
										{
											Type: "image_url",
											ImageURL: schedulingtypes.ImageBlock{
												Url: "data:image/png;base64,invalid!!!",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			wantEndpoints: false,
		},
		{
			name: "image URL without data prefix",
			request: &schedulingtypes.LLMRequest{
				Body: &schedulingtypes.LLMRequestBody{
					ChatCompletions: &schedulingtypes.ChatCompletionsRequest{
						Messages: []schedulingtypes.Message{
							{
								Content: schedulingtypes.Content{
									Structured: []schedulingtypes.ContentBlock{
										{
											Type: "image_url",
											ImageURL: schedulingtypes.ImageBlock{
												Url: "http://example.com/image.png",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			wantEndpoints: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			endpoint := schedulingtypes.NewEndpoint(nil, nil, nil)
			endpoints := []schedulingtypes.Endpoint{endpoint}

			err := plugin.PrepareRequestData(ctx, tt.request, endpoints)
			if err != nil {
				t.Fatalf("PrepareRequestData failed: %v", err)
			}

			val, ok := endpoint.Get(attrmm.MultimodalDataKey)
			if tt.wantEndpoints {
				if !ok {
					t.Fatalf("expected multimodal data in endpoint, but not found")
				}
				mmData := val.(*attrmm.MultimodalData)
				if !reflect.DeepEqual(mmData.Items, tt.wantItems) {
					t.Errorf("got items %v, want %v", mmData.Items, tt.wantItems)
				}
			} else {
				if ok {
					t.Errorf("expected no multimodal data in endpoint, but found")
				}
			}
		})
	}
}

func TestMultimodalEncoderPlugin_TypedName(t *testing.T) {
	p := &MultimodalEncoderPlugin{}
	got := p.TypedName()
	want := "multimodal-encoder-data-producer"
	if got.Name != want || got.Type != want {
		t.Errorf("TypedName() = %v, want %v", got, want)
	}
}

func TestMultimodalEncoderPlugin_ProducesConsumes(t *testing.T) {
	p := &MultimodalEncoderPlugin{}
	produces := p.Produces()
	if _, ok := produces[attrmm.MultimodalDataKey]; !ok {
		t.Errorf("Produces() should contain %s", attrmm.MultimodalDataKey)
	}

	consumes := p.Consumes()
	if len(consumes) != 0 {
		t.Errorf("Consumes() should be empty, got %v", consumes)
	}
}
