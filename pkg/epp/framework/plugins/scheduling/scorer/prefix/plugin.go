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

package prefix

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// Plugin implements the prefix cache aware scoring logic.
type Plugin struct {
	typedName plugin.TypedName
}

// compile-time type assertions
var (
	_ framework.Scorer = &Plugin{}
)

// PrefixCachePluginFactory defines the factory function for the Prefix plugin.
func PrefixCachePluginFactory(name string, _ json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	p, err := New(handle.Context())
	if err != nil {
		return nil, err
	}
	if name != "" {
		p = p.WithName(name)
	}
	return p, nil
}

// New initializes a new prefix Plugin.
func New(_ context.Context) (*Plugin, error) {
	return &Plugin{
		typedName: plugin.TypedName{
			Type: attrprefix.PrefixCachePluginType,
			Name: attrprefix.PrefixCachePluginType,
		},
	}, nil
}

// TypedName returns the type and name of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Category returns the preference the scorer applies (Affinity).
func (p *Plugin) Category() framework.ScorerCategory {
	return framework.Affinity
}

// WithName sets the name of the plugin instance.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// Produces returns the data produced by the plugin.
func (p *Plugin) Produces() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// Consumes returns the data consumed by the plugin.
func (p *Plugin) Consumes() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// Score returns the scoring result for the given list of pods based on prefix cache match info.
func (p *Plugin) Score(ctx context.Context, _ *framework.CycleState, _ *framework.LLMRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)

	for _, endpoint := range endpoints {
		info, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
		if !ok {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo not found for endpoint, assigning score 0", "endpoint", endpoint)
			scores[endpoint] = 0.0
			continue
		}

		if prefixMatchInfo, ok := info.(*attrprefix.PrefixCacheMatchInfo); ok {
			if prefixMatchInfo.TotalBlocks() == 0 {
				scores[endpoint] = 0.0
			} else {
				scores[endpoint] = float64(prefixMatchInfo.MatchBlocks()) / float64(prefixMatchInfo.TotalBlocks())
			}
		} else {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo has unexpected type, assigning score 0", "endpoint", endpoint)
			scores[endpoint] = 0.0
		}
	}
	return scores
}
<<<<<<< HEAD

// PreRequest records in the shared indexer the result of the scheduling selection.
// It updates the indexer with the prefix hashes for the selected endpoint(s).
func (p *Plugin) PreRequest(ctx context.Context, request *framework.LLMRequest, schedulingResult *framework.SchedulingResult) {
	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if len(primaryProfileResult.TargetEndpoints) == 0 {
		return
	}

	targetEndpoint := primaryProfileResult.TargetEndpoints[0]
	servers := []prepdataprefix.Server{p.makeServer(targetEndpoint)}

	// Also record for prefill node if present in P/D disaggregated mode.
	if pr, exists := schedulingResult.ProfileResults[prepdataprefix.Experimental_DefaultPrefillProfile]; exists && len(pr.TargetEndpoints) > 0 {
		servers = append(servers, p.makeServer(pr.TargetEndpoints[0]))
	}

	// Read state saved during PrepareRequestData.
	state, err := plugin.ReadPluginStateKey[*prepdataprefix.SchedulingContextState](p.pluginState, request.RequestId, plugin.StateKey(attrprefix.PrefixCachePluginType))
	p.pluginState.Delete(request.RequestId)
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to read prefix plugin state", "requestID", request.RequestId)
		return
	}

	// Update indexer asynchronously to avoid blocking the request path.
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		for _, s := range servers {
			p.indexer.Add(state.PrefixHashes, s)
		}
	}()

	// Record metrics.
	total := len(state.PrefixHashes)
	matchLen := state.PrefixCacheServers[prepdataprefix.ServerID(targetEndpoint.GetMetadata().NamespacedName)]
	blockSize := p.GetBlockSize(primaryProfileResult.TargetEndpoints)
	avgChars := prepdataprefix.AverageCharactersPerToken()

	metrics.RecordPrefixCacheMatch(matchLen*blockSize*avgChars, total*blockSize*avgChars)
}

func (p *Plugin) makeServer(targetEndpoint framework.Endpoint) prepdataprefix.Server {
	gpuBlocks := prepdataprefix.DefaultLRUCapacityPerServer
	if p.config.AutoTune && targetEndpoint.GetMetrics().CacheNumGPUBlocks > 0 {
		gpuBlocks = targetEndpoint.GetMetrics().CacheNumGPUBlocks
	}
	return prepdataprefix.Server{
		ServerID:       prepdataprefix.ServerID(targetEndpoint.GetMetadata().NamespacedName),
		NumOfGPUBlocks: gpuBlocks,
	}
}

// CleanUpInactivePods starts a goroutine that periodically removes inactive pods from the indexer.
func (p *Plugin) CleanUpInactivePods(ctx context.Context, handle plugin.Handle) {
	ticker := time.NewTicker(prepdataprefix.PodActiveCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			podNames := handle.PodList()
			activePods := make(map[prepdataprefix.ServerID]struct{}, len(podNames))
			for _, nsn := range podNames {
				activePods[prepdataprefix.ServerID(nsn)] = struct{}{}
			}

			for _, pod := range p.indexer.Pods() {
				if _, ok := activePods[pod]; !ok {
					p.indexer.RemovePod(pod)
					log.FromContext(ctx).V(logutil.VERBOSE).Info("Removed pod not in active set", "pod", pod)
				}
			}
		}
	}
}

// GetBlockSize returns the block size in tokens, potentially auto-tuned from endpoint metrics.
func (p *Plugin) GetBlockSize(endpoints []framework.Endpoint) int {
	if !p.config.AutoTune || len(endpoints) == 0 {
		return p.config.BlockSizeTokens
	}

<<<<<<< HEAD
	userInput, err := getUserInputBytes(request)
	if err != nil {
		loggerDebug.Error(err, "Failed to get user input bytes")
		return nil
	}

	// convert block size from tokens to characters
	cacheBlockSizeChars := blockSizeTokens * averageCharactersPerToken

	if len(userInput) < cacheBlockSizeChars {
		loggerDebug.Info("Request body too small for prefix cache", "size", len(userInput), "block size in chars", cacheBlockSizeChars)
		return nil
	}
	if len(userInput) > cacheBlockSizeChars*maxPrefixBlocks {
		loggerDebug.Info("Truncating input", "size", len(userInput), "max prefix blocks", maxPrefixBlocks, "block size in chars", cacheBlockSizeChars)
		userInput = userInput[:maxPrefixBlocks*cacheBlockSizeChars]
	}
	// Split the body into blocks of size cacheBlockSize.
	// If the last block is smaller than cacheBlockSize, it will be ignored.
	res := make([]BlockHash, 0, len(userInput)/cacheBlockSizeChars)
	// Add the model to the first block hash so that different models have different hashes even with the same body.
	h := xxhash.New()
	_, _ = h.Write([]byte(request.TargetModel))
	if cacheSalt := request.Body.CacheSalt(); cacheSalt != "" {
		_, _ = h.Write([]byte(cacheSalt))
	}

	prevBlockHash := BlockHash(h.Sum64())
	for i := 0; i+cacheBlockSizeChars <= len(userInput); i += cacheBlockSizeChars {
		h.Reset()
		_, _ = h.Write(userInput[i : i+cacheBlockSizeChars])
		_, _ = h.Write(toBytes(prevBlockHash))
		res = append(res, BlockHash(h.Sum64()))

		prevBlockHash = res[len(res)-1]
	}

	return res
}

func toBytes(i BlockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
}

func getUserInputBytes(request *framework.LLMRequest) ([]byte, error) {
	switch {
	case request.Body.Conversations != nil:
		// Handle conversations API - marshal the entire items slice for cache key generation
		return json.Marshal(request.Body.Conversations.Items)

	case request.Body.Responses != nil:
		// Handle responses API - use ordered slice to ensure deterministic marshaling
		var combined []map[string]interface{}

		// 1. Instructions (if present)
		if request.Body.Responses.Instructions != nil {
			combined = append(combined, map[string]interface{}{
				"instructions": request.Body.Responses.Instructions,
			})
		}

		// 2. Tools (if present)
		if request.Body.Responses.Tools != nil {
			combined = append(combined, map[string]interface{}{
				"tools": request.Body.Responses.Tools,
			})
		}

		// 3. Input (always present)
		combined = append(combined, map[string]interface{}{
			"input": request.Body.Responses.Input,
		})

		return json.Marshal(combined)

	case request.Body.ChatCompletions != nil:
		// Handle chat completions API (maintain backward compatibility)
		return json.Marshal(request.Body.ChatCompletions.Messages)

	case request.Body.Completions != nil:
		// Handle completions API (maintain backward compatibility)
		return []byte(request.Body.Completions.Prompt), nil

	case request.Body.Embeddings != nil:
		// Handle embeddings API - marshal input for cache key generation
		return json.Marshal(request.Body.Embeddings.Input)

	default:
		return nil, errors.New("invalid request body: no recognized API format found")
	}
}

// GetBlockSize returns the block size in tokens.
// In case of auto-tune uses the block size from the first endpoint, otherwise uses the block size from the configuration
func GetBlockSize(endpoints []framework.Endpoint, config Config) int {
	if !config.AutoTune {
		return config.BlockSizeTokens
	}

	// Fallback to BlockSize if no metrics are available.
	if len(endpoints) == 0 {
		return config.BlockSizeTokens
	}

	// Since all Endpoints originate from the same inference pool, they are considered to have identical configurations.
	// Therefore, using the CacheBlockSize value from the first Endpoint suffices.
=======
>>>>>>> ba020db4 (Migrate data preparation code into the preparedata directory in a new plugin)
	if endpoint := endpoints[0]; endpoint.GetMetrics() != nil {
		cacheBlockSize := endpoint.GetMetrics().CacheBlockSize
		if cacheBlockSize > 0 {
			return cacheBlockSize
		}
	}
	return p.config.BlockSizeTokens
}
=======
>>>>>>> 567bd0be (move indexer out, also inject metrics exported into preparedata to set the code pattern like a framework)
