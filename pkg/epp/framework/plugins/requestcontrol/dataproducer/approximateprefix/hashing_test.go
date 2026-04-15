package approximateprefix

import (
	"encoding/json"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)



func TestHashMultimodalBlock(t *testing.T) {
	tests := []struct {
		name    string
		block   requesthandling.ContentBlock
		wantErr bool
	}{
		{
			name: "Image block",
			block: requesthandling.ContentBlock{
				Type: "image_url",
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
								Role: "user",
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
				var combined [][]string
				if err := json.Unmarshal(got, &combined); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(combined) != 1 {
					t.Fatalf("Expected 1 element, got %d", len(combined))
				}
				msg := combined[0]
				if len(msg) != 3 || msg[0] != "user" || msg[1] != "Hello" || msg[2] != "World" {
					t.Errorf("Unexpected content: %v", msg)
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
								Role: "user",
								Content: requesthandling.Content{
									Structured: []requesthandling.ContentBlock{
										{Type: "text", Text: "Describe:"},
										{Type: "image_url", ImageURL: requesthandling.ImageBlock{Url: "url1"}},
									},
								},
							},
						},
					},
				},
			},
			verify: func(t *testing.T, got []byte) {
				var combined [][]string
				if err := json.Unmarshal(got, &combined); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(combined) != 1 {
					t.Fatalf("Expected 1 element, got %d", len(combined))
				}
				msg := combined[0]
				if len(msg) != 3 {
					t.Fatalf("Expected 3 elements in message, got %d", len(msg))
				}
				if msg[0] != "user" {
					t.Errorf("Expected 'user', got %s", msg[0])
				}
				if msg[1] != "Describe:" {
					t.Errorf("Expected 'Describe:', got %s", msg[1])
				}
				if len(msg[2]) != 64 {
					t.Errorf("Expected 64-char hash, got length %d", len(msg[2]))
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
								Role: "user",
								Content: requesthandling.Content{
									Structured: []requesthandling.ContentBlock{
										{Type: "image_url", ImageURL: requesthandling.ImageBlock{Url: "url1"}},
										{Type: "image_url", ImageURL: requesthandling.ImageBlock{Url: "url2"}},
									},
								},
							},
						},
					},
				},
			},
			verify: func(t *testing.T, got []byte) {
				var combined [][]string
				if err := json.Unmarshal(got, &combined); err != nil {
					t.Fatalf("Failed to unmarshal: %v", err)
				}
				if len(combined) != 1 {
					t.Fatalf("Expected 1 element, got %d", len(combined))
				}
				msg := combined[0]
				if len(msg) != 3 {
					t.Fatalf("Expected 3 elements in message, got %d", len(msg))
				}
				if msg[0] != "user" {
					t.Errorf("Expected 'user', got %s", msg[0])
				}
				for i := 1; i < 3; i++ {
					if len(msg[i]) != 64 {
						t.Errorf("Element %d: expected 64-char hash, got length %d", i, len(msg[i]))
					}
				}
				if msg[1] == msg[2] {
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
