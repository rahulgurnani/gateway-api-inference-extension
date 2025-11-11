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

import "errors"

// buildDAG builds a dependency graph among data preparation plugins based on their
// produced and consumed data keys.
func buildDAG(plugins []PrepareDataPlugin) map[string][]string {
	dag := make(map[string][]string)
	for _, plugin := range plugins {
		dag[plugin.TypedName().String()] = []string{}
	}
	// Create dependency graph as a DAG.
	for i := range plugins {
		for j := range plugins {
			if i == j {
				continue
			}
			// Check whether plugin[i] produces something consumed by plugin[j]. In that case, j depends on i.
			if plugins[i].Produces() != nil && plugins[j].Consumes() != nil {
				// For all the keys produced by plugin i, check if plugin j consumes any of them.
				// If yes, then j depends on i.
				for producedKey := range plugins[i].Produces() {
					// If plugin j consumes the produced key, then j depends on i. We can break after the first match.
					if _, ok := plugins[j].Consumes()[producedKey]; ok {
						iPluginName := plugins[i].TypedName().String()
						jPluginName := plugins[j].TypedName().String()
						dag[jPluginName] = append(dag[jPluginName], iPluginName)
						break
					}
				}
			}
		}
	}
	return dag
}

// prepareDataGraph builds a DAG of data preparation plugins and checks for cycles.
// If there is a cycle, it returns an error.
func prepareDataGraph(plugins []PrepareDataPlugin) (map[string][]string, error) {
	dag := buildDAG(plugins)

	// Check for cycles in the DAG.
	// TODO: Perform the error validation on the startup.
	if cycleExistsInDAG(dag) {
		return nil, errors.New("cycle detected in data preparation plugin dependencies")
	}

	return dag, nil
}

// cycleExistsInDAG checks if there are cycles in the given directed graph represented as an adjacency list.
func cycleExistsInDAG(dag map[string][]string) bool {
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	var dfs func(string) bool
	dfs = func(node string) bool {
		if recStack[node] {
			return true // Cycle detected
		}
		if visited[node] {
			return false
		}
		visited[node] = true
		recStack[node] = true

		for _, neighbor := range dag[node] {
			if dfs(neighbor) {
				return true
			}
		}
		recStack[node] = false
		return false
	}

	for pluginName := range dag {
		if !visited[pluginName] {
			if dfs(pluginName) {
				return true
			}
		}
	}
	return false
}
