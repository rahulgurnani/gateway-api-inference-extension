// /*
// Copyright 2025 The Kubernetes Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// */

package scorer

import (
	"context"
	"encoding/json"

	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	PrefixCacheScorerType = "prefix-cache-scorer"
	PrefixCacheScoreKey   = "PrefixCacheScoreKey"
)

type ServerID k8stypes.NamespacedName

// compile-time type assertion
var _ framework.Scorer = &PrefixCacheScorer{}

// PrefixCacheScorerFactory defines the factory function for PrefixCacheScorer.
func PrefixCacheScorerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewPrefixCacheScorer().WithName(name), nil
}

// NewPrefixCacheScorer initializes a new PrefixCacheScorer and returns its pointer.
func NewPrefixCacheScorer() *PrefixCacheScorer {
	return &PrefixCacheScorer{
		tn: plugins.TypedName{Type: PrefixCacheScorerType, Name: PrefixCacheScorerType},
	}
}

// PrefixCacheScorer scores list of candidate pods based on Lora affinity and availability.
type PrefixCacheScorer struct {
	tn    plugins.TypedName
	state prefix.SchedulingContextState
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *PrefixCacheScorer) TypedName() plugins.TypedName {
	return s.tn
}

// Consumes returns the list of data that is consumed by the plugin.
func (s *PrefixCacheScorer) Consumes() map[string]any {
	return map[string]any{}
}

// WithName sets the name of the scorer.
func (s *PrefixCacheScorer) WithName(name string) *PrefixCacheScorer {
	s.tn.Name = name
	return s
}

func (s *PrefixCacheScorer) Score(_ context.Context, cycleState *types.CycleState, _ *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	// calculate the scores of pods
	scores := make(map[types.Pod]float64, len(pods))
	total := len(s.state.PrefixHashes)

	podScoreFunc := func(pod types.Pod) float64 {
		if total == 0 {
			return 0
		}
		// TODO: fix this later.
		matchLen := s.state.PrefixCacheServers[prefix.ServerID(pod.GetPod().NamespacedName)]
		return float64(matchLen) / float64(total)
	}

	for _, pod := range pods {
		scores[pod] = podScoreFunc(pod)
	}
	return scores
}
