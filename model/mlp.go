package model

import (
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

// MLPOpts holds configuration values for an MLP.
type MLPOpts struct {
	Activation Activation
	Source     randv2.Source
}

// MLPOptionFunc configures an MLP.
type MLPOptionFunc func(*MLP)

// WithMLPSource sets the random source used to initialize the model.
func WithMLPSource(s randv2.Source) MLPOptionFunc {
	return func(l *MLP) {
		l.s = s
	}
}

// WithMLPActivation sets the activation used between hidden layers.
func WithMLPActivation(activation Activation) MLPOptionFunc {
	return func(l *MLP) {
		l.Activation = activation
	}
}

// MLP is a multilayer perceptron.
type MLP struct {
	Activation Activation
	s          randv2.Source
	Model
}

// NewMLP returns a new multilayer perceptron with the given layer sizes.
func NewMLP(outSize []int, opts ...MLPOptionFunc) *MLP {
	mlp := &MLP{
		Activation: F.Sigmoid,
		Model: Model{
			Layers: make([]Layer, len(outSize)),
		},
	}

	for _, opt := range opts {
		opt(mlp)
	}

	for i := range outSize {
		mlp.Layers[i] = L.Linear(outSize[i], L.WithSource(mlp.s))
	}

	return mlp
}

// Forward applies the model to x and returns the output.
func (m *MLP) Forward(x *variable.Variable) *variable.Variable {
	last := len(m.Layers) - 1
	for _, l := range m.Layers[:last] {
		x = m.Activation(l.First(x))
	}

	return m.Layers[last].First(x)
}
