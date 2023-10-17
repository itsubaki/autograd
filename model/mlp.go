package model

import (
	"math/rand"

	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type MLPOpts struct {
	Activation Activation
	Source     rand.Source
}

type MLP struct {
	Activation Activation
	Model
}

func NewMLP(outSize []int, opts ...MLPOpts) *MLP {
	activation := F.Sigmoid
	if len(opts) > 0 && opts[0].Activation != nil {
		activation = opts[0].Activation
	}

	var s rand.Source
	if len(opts) > 0 && opts[0].Source != nil {
		s = opts[0].Source
	}

	layers := make([]Layer, len(outSize))
	for i := 0; i < len(outSize); i++ {
		layers[i] = L.Linear(outSize[i], L.LinearOpts{
			Source: s,
		})
	}

	return &MLP{
		Activation: activation,
		Model: Model{
			Layers: layers,
		},
	}
}

func (m *MLP) Forward(x *variable.Variable) *variable.Variable {
	last := len(m.Layers) - 1
	for _, l := range m.Layers[:last] {
		x = m.Activation(l.First(x))
	}

	return m.Layers[last].First(x)
}
