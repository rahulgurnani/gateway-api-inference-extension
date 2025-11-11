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

package requestcontrol

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

type mockPrepareDataPlugin struct {
	name     string
	produces map[string]any
	consumes map[string]any
}

func (m *mockPrepareDataPlugin) TypedName() plugins.TypedName {
	return plugins.TypedName{Name: m.name, Type: "mock"}
}

func (m *mockPrepareDataPlugin) Produces() map[string]any {
	return m.produces
}

func (m *mockPrepareDataPlugin) Consumes() map[string]any {
	return m.consumes
}

func (m *mockPrepareDataPlugin) PrepareRequestData(ctx context.Context, request *types.LLMRequest, pods []types.Pod) error {
	pods[0].Put(mockProducedDataKey, mockProducedDataType{value: 42})
	return nil
}

func TestPrepareDataGraph(t *testing.T) {
	pluginA := &mockPrepareDataPlugin{name: "A", produces: map[string]any{"keyA": nil}}
	pluginB := &mockPrepareDataPlugin{name: "B", consumes: map[string]any{"keyA": nil}, produces: map[string]any{"keyB": nil}}
	pluginC := &mockPrepareDataPlugin{name: "C", consumes: map[string]any{"keyB": nil}}
	pluginD := &mockPrepareDataPlugin{name: "D", consumes: map[string]any{"keyA": nil}}
	pluginE := &mockPrepareDataPlugin{name: "E"} // No dependencies

	// Cycle plugins
	pluginX := &mockPrepareDataPlugin{name: "X", produces: map[string]any{"keyX": nil}, consumes: map[string]any{"keyY": nil}}
	pluginY := &mockPrepareDataPlugin{name: "Y", produces: map[string]any{"keyY": nil}, consumes: map[string]any{"keyX": nil}}

	testCases := []struct {
		name        string
		plugins     []PrepareDataPlugin
		expectedDAG map[string][]string
		expectError bool
	}{
		{
			name:        "No plugins",
			plugins:     []PrepareDataPlugin{},
			expectedDAG: map[string][]string{},
			expectError: false,
		},
		{
			name:    "Plugins with no dependencies",
			plugins: []PrepareDataPlugin{pluginA, pluginE},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"E/mock": {},
			},
			expectError: false,
		},
		{
			name:    "Simple linear dependency (A -> B -> C)",
			plugins: []PrepareDataPlugin{pluginA, pluginB, pluginC},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"B/mock": {"A/mock"},
				"C/mock": {"B/mock"},
			},
			expectError: false,
		},
		{
			name:    "DAG with multiple dependencies (A -> B, A -> D)",
			plugins: []PrepareDataPlugin{pluginA, pluginB, pluginD, pluginE},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"B/mock": {"A/mock"},
				"D/mock": {"A/mock"},
				"E/mock": {},
			},
			expectError: false,
		},
		{
			name:        "Graph with a cycle (X -> Y, Y -> X)",
			plugins:     []PrepareDataPlugin{pluginX, pluginY},
			expectedDAG: nil,
			expectError: true,
		},
		{
			name:        "Complex graph with a cycle",
			plugins:     []PrepareDataPlugin{pluginA, pluginB, pluginX, pluginY},
			expectedDAG: nil,
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dag, err := prepareDataGraph(tc.plugins)

			if tc.expectError {
				assert.Error(t, err)
				assert.Nil(t, dag)
				assert.Contains(t, err.Error(), "cycle detected")
			} else {
				assert.NoError(t, err)

				// Normalize the slices in the maps for consistent comparison
				normalizedDAG := make(map[string][]string)
				for k, v := range dag {
					normalizedDAG[k] = v
				}
				normalizedExpectedDAG := make(map[string][]string)
				for k, v := range tc.expectedDAG {
					normalizedExpectedDAG[k] = v
				}

				if diff := cmp.Diff(normalizedExpectedDAG, normalizedDAG); diff != "" {
					t.Errorf("prepareDataGraph() mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}
