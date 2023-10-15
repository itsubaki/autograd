package model

import (
	"math/rand"

	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type MLP struct {
	Model
	Activation Activation
}

type MLPOpts struct {
	Activation Activation
	Source     rand.Source
}

func NewMLP(outSize []int, opts ...MLPOpts) *MLP {
	activation := F.Sigmoid
	if len(opts) > 0 && opts[0].Activation != nil {
		activation = opts[0].Activation
	}

	s := make([]rand.Source, 0)
	if len(opts) > 0 && opts[0].Source != nil {
		s = append(s, opts[0].Source)
	}

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
