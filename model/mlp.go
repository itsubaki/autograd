package model

import (
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type MLPOpts struct {
	Activation Activation
	Source     randv2.Source
}

type MLPOptionFunc func(*MLP)

func WithMLPSource(s randv2.Source) MLPOptionFunc {
	return func(l *MLP) {
		l.s = s
	}
}

func WithMLPActivation(activation Activation) MLPOptionFunc {
	return func(l *MLP) {
		l.Activation = activation
	}
}

type MLP struct {
	Activation Activation
	s          randv2.Source
	Model
}

func NewMLP(outSize []int, opts ...MLPOptionFunc) *MLP {
	mlp := &MLP{
		Activation: F.Sigmoid,
		Model: Model{
			Layers: make([]L.Layer, len(outSize)),
		},
	}

	for _, opt := range opts {
		opt(mlp)
	}

	for i := 0; i < len(outSize); i++ {
		mlp.Layers[i] = L.Linear(outSize[i], L.WithSource(mlp.s))
	}

	return mlp
}

func (m *MLP) Forward(x *variable.Variable) *variable.Variable {
	last := len(m.Layers) - 1
	for _, l := range m.Layers[:last] {
		x = m.Activation(l.First(x))
	}

	return m.Layers[last].First(x)
}
