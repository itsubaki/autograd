package model

import (
	randv2 "math/rand/v2"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

// LSTMOptionFunc configures an LSTM model.
type LSTMOptionFunc func(*LSTM)

// WithLSTMSource sets the random source used to initialize the model.
func WithLSTMSource(s randv2.Source) LSTMOptionFunc {
	return func(l *LSTM) {
		l.s = s
	}
}

// LSTM is an LSTM-based sequence model followed by a linear output layer.
type LSTM struct {
	s randv2.Source
	Model
}

// NewLSTM returns a new LSTM model.
func NewLSTM(hiddenSize, outSize int, opts ...LSTMOptionFunc) *LSTM {
	lstm := &LSTM{
		Model: Model{
			Layers: make([]Layer, 0),
		},
	}

	for _, opt := range opts {
		opt(lstm)
	}

	lstm.Layers = append(lstm.Layers, []Layer{
		L.LSTM(hiddenSize, L.WithLSTMSource(lstm.s)),
		L.Linear(outSize, L.WithSource(lstm.s)),
	}...)

	return lstm
}

// ResetState clears the hidden and cell states of the LSTM layer.
func (m *LSTM) ResetState() {
	m.Layers[0].(*L.LSTMT).ResetState()
}

// Forward applies the model to x and returns the output.
func (m *LSTM) Forward(x *variable.Variable) *variable.Variable {
	for _, l := range m.Layers {
		x = l.First(x)
	}

	return x
}
