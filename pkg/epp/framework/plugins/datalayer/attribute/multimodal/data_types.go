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

package multimodal

import (
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

const (
	MultimodalDataKey = "MultimodalDataKey"
)

// MultimodalItem holds the decoded bytes and content hash for a single multimodal element.
type MultimodalItem struct {
	Data []byte
	Hash string
}

// MultimodalData holds all multimodal items extracted from a request.
type MultimodalData struct {
	Items []MultimodalItem
}

func (m *MultimodalData) Clone() fwkdl.Cloneable {
	if m == nil {
		return nil
	}
	cloned := &MultimodalData{
		Items: make([]MultimodalItem, len(m.Items)),
	}
	for i, item := range m.Items {
		dataCopy := make([]byte, len(item.Data))
		copy(dataCopy, item.Data)
		cloned.Items[i] = MultimodalItem{
			Data: dataCopy,
			Hash: item.Hash,
		}
	}
	return cloned
}
