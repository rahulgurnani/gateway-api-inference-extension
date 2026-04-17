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
	"encoding/json"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const userRole = "user"

func TestHashMultimodalContent(t *testing.T) {
	tests := []struct {
		name    string
		block   requesthandling.ContentBlock
		wantErr bool
	}{
		{
			name: "Image block",
			block: requesthandling.ContentBlock{
				Type: imageURLType,
				ImageURL: requesthandling.ImageBlock{
					Url: "data:image/png;base64,...",
				},
			},
		},
		{
			name: "Text block",
			block: requesthandling.ContentBlock{
				Type: "text",
				Text: "Hello",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := hashMultimodalContent(tt.block)
			if (err != nil) != tt.wantErr {
				t.Errorf("hashMultimodalContent() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got == "" && !tt.wantErr {
				t.Errorf("hashMultimodalContent() got empty string")
			}
			if len(got) != 16 && !tt.wantErr {
				t.Errorf("hashMultimodalContent() got length %d, want 16", len(got))
			}
		})
	}
}

func TestGetUserInputBytes_ChatCompletions(t *testing.T) {
	tests := []struct {
		name    string
		request *scheduling.InferenceRequest
		verify  func(t *testing.T, got []byte)
		wantErr bool
	}{
		{
			name: "Only text content",
			request: &scheduling.InferenceRequest{
				Body: &requesthandling.InferenceRequestBody{
					ChatCompletions: &requesthandling.ChatCompletionsRequest{
						Messages: []requesthandling.Message{
							{
								Role: userRole,
								Content: requesthandling.Content{
									Structured: []requesthandling.ContentBlock{
										{Type: "text", Text: "Hello"},
										{Type: "text", Text: "World"},
									},
								},
							},
						},
					},
				},
			},
			verify: func(t *testing.T, got []byte) {
				var messages []requesthandling.Message
				if err := json.Unmarshal(got, &messages); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(messages) != 1 {
					t.Fatalf("Expected 1 message, got %d", len(messages))
				}
				if messages[0].Role != userRole {
					t.Errorf("Expected role %q, got %q", userRole, messages[0].Role)
				}
				blocks := messages[0].Content.Structured
				if len(blocks) != 2 || blocks[0].Text != "Hello" || blocks[1].Text != "World" {
					t.Errorf("Unexpected content blocks: %v", blocks)
				}
			},
		},
		{
			name: "Mixed content",
			request: &scheduling.InferenceRequest{
				Body: &requesthandling.InferenceRequestBody{
					ChatCompletions: &requesthandling.ChatCompletionsRequest{
						Messages: []requesthandling.Message{
							{
								Role: userRole,
								Content: requesthandling.Content{
									Structured: []requesthandling.ContentBlock{
										{Type: "text", Text: "Describe:"},
										{Type: imageURLType,
											ImageURL: requesthandling.ImageBlock{Url: "url1"}},
									},
								},
							},
						},
					},
				},
			},
			verify: func(t *testing.T, got []byte) {
				var messages []requesthandling.Message
				if err := json.Unmarshal(got, &messages); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(messages) != 1 {
					t.Fatalf("Expected 1 message, got %d", len(messages))
				}
				if messages[0].Role != userRole {
					t.Errorf("Expected role %q, got %q", userRole, messages[0].Role)
				}
				blocks := messages[0].Content.Structured
				if len(blocks) != 2 {
					t.Fatalf("Expected 2 blocks, got %d", len(blocks))
				}
				if blocks[0].Text != "Describe:" {
					t.Errorf("Expected 'Describe:', got %q", blocks[0].Text)
				}
				if len(blocks[1].ImageURL.Url) != 16 {
					t.Errorf("Expected 16-char hash URL, got length %d", len(blocks[1].ImageURL.Url))
				}
			},
		},
		{
			name: "Long Google Storage URL",
			request: &scheduling.InferenceRequest{
				Body: &requesthandling.InferenceRequestBody{
					ChatCompletions: &requesthandling.ChatCompletionsRequest{
						Messages: []requesthandling.Message{
							{
								Role: userRole,
								Content: requesthandling.Content{
									Structured: []requesthandling.ContentBlock{
										{Type: "text", Text: "Analyze this image:"},
										{Type: imageURLType,
											ImageURL: requesthandling.ImageBlock{Url: "https://storage.googleapis.com/averylargesizednameofabuckettostoreimages/sample52.jpg"}},
									},
								},
							},
						},
					},
				},
			},
			verify: func(t *testing.T, got []byte) {
				var messages []requesthandling.Message
				if err := json.Unmarshal(got, &messages); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(messages) != 1 {
					t.Fatalf("Expected 1 message, got %d", len(messages))
				}
				if messages[0].Role != userRole {
					t.Errorf("Expected role %q, got %q", userRole, messages[0].Role)
				}
				blocks := messages[0].Content.Structured
				if len(blocks) != 2 {
					t.Fatalf("Expected 2 blocks, got %d", len(blocks))
				}
				if blocks[0].Text != "Analyze this image:" {
					t.Errorf("Expected 'Analyze this image:', got %q", blocks[0].Text)
				}
				if len(blocks[1].ImageURL.Url) != 16 {
					t.Errorf("Expected 16-char hash URL, got length %d", len(blocks[1].ImageURL.Url))
				}
			},
		},
		{
			name: "Only image content",
			request: &scheduling.InferenceRequest{
				Body: &requesthandling.InferenceRequestBody{
					ChatCompletions: &requesthandling.ChatCompletionsRequest{
						Messages: []requesthandling.Message{
							{
								Role: userRole,
								Content: requesthandling.Content{
									Structured: []requesthandling.ContentBlock{
										{Type: imageURLType,
											ImageURL: requesthandling.ImageBlock{Url: "url1"}},
										{Type: imageURLType,
											ImageURL: requesthandling.ImageBlock{Url: "url2"}},
									},
								},
							},
						},
					},
				},
			},
			verify: func(t *testing.T, got []byte) {
				var messages []requesthandling.Message
				if err := json.Unmarshal(got, &messages); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(messages) != 1 {
					t.Fatalf("Expected 1 message, got %d", len(messages))
				}
				blocks := messages[0].Content.Structured
				if len(blocks) != 2 {
					t.Fatalf("Expected 2 blocks, got %d", len(blocks))
				}
				for i, block := range blocks {
					if len(block.ImageURL.Url) != 16 {
						t.Errorf("Block %d: expected 16-char hash URL, got length %d", i, len(block.ImageURL.Url))
					}
				}
				if blocks[0].ImageURL.Url == blocks[1].ImageURL.Url {
					t.Errorf("Hashes for different URLs should be different")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := getUserInputBytes(tt.request)
			if (err != nil) != tt.wantErr {
				t.Errorf("getUserInputBytes() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if tt.verify != nil {
				tt.verify(t, got)
			}
		})
	}
}
