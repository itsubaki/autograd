package model

import (
	"math/rand"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type MLP struct {
	Model
	Activation Activation
}

func NewMLP(outSize []int, activation Activation, s ...rand.Source) *MLP {
	layers := make([]*L.Layer, len(outSize))
	for i := 0; i < len(outSize); i++ {
		layers[i] = L.Linear(outSize[i], s...)
	}

	return &MLP{
		Model: Model{
			Layer: L.Layer{
				Layers: layers,
			},
		},
		Activation: activation,
	}
}

func (m *MLP) Forward(x *variable.Variable) *variable.Variable {
	last := len(m.Layers) - 1
	for _, l := range m.Layers[:last] {
		x = m.Activation(l.First(x))
	}

	return m.Layers[last].First(x)
}
