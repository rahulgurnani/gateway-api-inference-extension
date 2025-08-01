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

package types

import (
	"errors"
	"fmt"
	"sync"
)

var (
	// ErrNotFound is the not found error message.
	ErrNotFound = errors.New("not found")
)

// StateData is a generic type for arbitrary data stored in CycleState.
type StateData interface {
	// Clone is an interface to make a copy of StateData.
	Clone() StateData
}

// StateKey is the type of keys stored in CycleState.
type StateKey string

// NewCycleState initializes a new CycleState and returns its pointer.
func NewCycleState() *CycleState {
	return &CycleState{}
}

// CycleState provides a mechanism for plugins to store and retrieve arbitrary data.
// StateData stored by one plugin can be read, altered, or deleted by another plugin.
// CycleState does not provide any data protection, as all plugins are assumed to be
// trusted.
// Note: CycleState uses a sync.Map to back the storage, because it is thread safe. It's aimed to optimize for the "write once and read many times" scenarios.
type CycleState struct {
	// key: StateKey, value: StateData
	storage sync.Map
}

// Read retrieves data with the given "key" from CycleState. If the key is not
// present, ErrNotFound is returned.
//
// See CycleState for notes on concurrency.
func (c *CycleState) Read(key StateKey) (StateData, error) {
	if v, ok := c.storage.Load(key); ok {
		return v.(StateData), nil
	}
	return nil, ErrNotFound
}

// Write stores the given "val" in CycleState with the given "key".
//
// See CycleState for notes on concurrency.
func (c *CycleState) Write(key StateKey, val StateData) {
	c.storage.Store(key, val)
}

// Delete deletes data with the given key from CycleState.
//
// See CycleState for notes on concurrency.
func (c *CycleState) Delete(key StateKey) {
	c.storage.Delete(key)
}

// ReadCycleStateKey  retrieves data with the given key from CycleState and asserts it to type T.
// Returns an error if the key is not found or the type assertion fails.
func ReadCycleStateKey[T StateData](c *CycleState, key StateKey) (T, error) {
	var zero T

	raw, err := c.Read(key)
	if err != nil {
		return zero, err
	}

	val, ok := raw.(T)
	if !ok {
		return zero, fmt.Errorf("unexpected type for key %q: got %T", key, raw)
	}

	return val, nil
}
