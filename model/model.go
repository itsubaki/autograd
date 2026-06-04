package model

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var (
	_ Layer = (*L.LinearT)(nil)
	_ Layer = (*L.RNNT)(nil)
	_ Layer = (*L.LSTMT)(nil)
)

// Layer is the interface implemented by trainable model layers.
type Layer interface {
	First(x ...*variable.Variable) *variable.Variable
	Forward(x ...*variable.Variable) []*variable.Variable
	Params() L.Parameters
	Cleargrads()
}

// Model represents a stack of layers.
type Model struct {
	Layers []string
	L      map[string]Layer
}

func (m *Model) Add(name string, layer Layer) {
	if m.Layers == nil {
		m.Layers = make([]string, 0)
	}

	if m.L == nil {
		m.L = make(map[string]Layer)
	}

	m.Layers = append(m.Layers, name)
	m.L[name] = layer
}

// Params returns all parameters in the model keyed by layer index and parameter name.
func (m *Model) Params() L.Parameters {
	params := make(L.Parameters, 0)
	for name, layer := range m.L {
		for k, p := range layer.Params() {
			params[fmt.Sprintf("%s.%s", name, k)] = p
		}
	}

	return params
}

// Cleargrads clears the gradients of all model parameters.
func (m *Model) Cleargrads() {
	for _, l := range m.L {
		l.Cleargrads()
	}
}
