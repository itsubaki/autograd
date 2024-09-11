package model

import (
	randv2 "math/rand/v2"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type LSTMOptionFunc func(*LSTM)

func WithLSTMSource(s randv2.Source) LSTMOptionFunc {
	return func(l *LSTM) {
		l.s = s
	}
}

type LSTM struct {
	s randv2.Source
	Model
}

func NewLSTM(hiddenSize, outSize int, opts ...LSTMOptionFunc) *LSTM {
	lstm := &LSTM{
		Model: Model{
			Layers: make([]L.Layer, 0),
		},
	}

	for _, opt := range opts {
		opt(lstm)
	}

	lstm.Layers = append(lstm.Layers, []L.Layer{
		L.LSTM(hiddenSize, L.WithLSTMSource(lstm.s)),
		L.Linear(outSize, L.WithSource(lstm.s)),
	}...)

	return lstm
}

func (m *LSTM) ResetState() {
	m.Layers[0].(*L.LSTMT).ResetState()
}

func (m *LSTM) Forward(x *variable.Variable) *variable.Variable {
	for _, l := range m.Layers {
		x = l.First(x)
	}

	return x
}
