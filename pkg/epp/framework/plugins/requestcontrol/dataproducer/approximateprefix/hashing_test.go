package approximateprefix

import (
	"encoding/json"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)



func TestHashMultimodalBlock(t *testing.T) {
	tests := []struct {
		name    string
		block   scheduling.ContentBlock
		wantErr bool
	}{
		{
			name: "Image block",
			block: scheduling.ContentBlock{
				Type: "image_url",
				ImageURL: scheduling.ImageBlock{
					Url: "data:image/png;base64,...",
				},
			},
		},
		{
			name: "Text block",
			block: scheduling.ContentBlock{
				Type: "text",
				Text: "Hello",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := hashMultimodalBlock(tt.block)
			if (err != nil) != tt.wantErr {
				t.Errorf("hashMultimodalBlock() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got == "" && !tt.wantErr {
				t.Errorf("hashMultimodalBlock() got empty string")
			}
			if len(got) != 64 && !tt.wantErr {
				t.Errorf("hashMultimodalBlock() got length %d, want 64", len(got))
			}
		})
	}
}

func TestGetUserInputBytes_ChatCompletions(t *testing.T) {
	tests := []struct {
		name    string
		request *scheduling.LLMRequest
		verify  func(t *testing.T, got []byte)
		wantErr bool
	}{
		{
			name: "Only text content",
			request: &scheduling.LLMRequest{
				Body: &scheduling.LLMRequestBody{
					ChatCompletions: &scheduling.ChatCompletionsRequest{
						Messages: []scheduling.Message{
							{
								Content: scheduling.Content{
									Structured: []scheduling.ContentBlock{
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
				var combined []string
				if err := json.Unmarshal(got, &combined); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(combined) != 2 || combined[0] != "Hello" || combined[1] != "World" {
					t.Errorf("Unexpected content: %v", combined)
				}
			},
		},
		{
			name: "Mixed content",
			request: &scheduling.LLMRequest{
				Body: &scheduling.LLMRequestBody{
					ChatCompletions: &scheduling.ChatCompletionsRequest{
						Messages: []scheduling.Message{
							{
								Content: scheduling.Content{
									Structured: []scheduling.ContentBlock{
										{Type: "text", Text: "Describe:"},
										{Type: "image_url", ImageURL: scheduling.ImageBlock{Url: "url1"}},
									},
								},
							},
						},
					},
				},
			},
			verify: func(t *testing.T, got []byte) {
				var combined []string
				if err := json.Unmarshal(got, &combined); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(combined) != 2 {
					t.Fatalf("Expected 2 elements, got %d", len(combined))
				}
				if combined[0] != "Describe:" {
					t.Errorf("Expected 'Describe:', got %s", combined[0])
				}
				if len(combined[1]) != 64 {
					t.Errorf("Expected 64-char hash, got length %d", len(combined[1]))
				}
			},
		},
		{
			name: "Only image content",
			request: &scheduling.LLMRequest{
				Body: &scheduling.LLMRequestBody{
					ChatCompletions: &scheduling.ChatCompletionsRequest{
						Messages: []scheduling.Message{
							{
								Content: scheduling.Content{
									Structured: []scheduling.ContentBlock{
										{Type: "image_url", ImageURL: scheduling.ImageBlock{Url: "url1"}},
										{Type: "image_url", ImageURL: scheduling.ImageBlock{Url: "url2"}},
									},
								},
							},
						},
					},
				},
			},
			verify: func(t *testing.T, got []byte) {
				var combined []string
				if err := json.Unmarshal(got, &combined); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(combined) != 2 {
					t.Fatalf("Expected 2 elements, got %d", len(combined))
				}
				for i, hash := range combined {
					if len(hash) != 64 {
						t.Errorf("Element %d: expected 64-char hash, got length %d", i, len(hash))
					}
				}
				if combined[0] == combined[1] {
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
