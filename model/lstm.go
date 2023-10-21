package model

import (
	"math/rand"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type LSTMOpts struct {
	Source rand.Source
}

type LSTM struct {
	Model
}

func NewLSTM(hiddenSize, outSize int, opts ...LSTMOpts) *LSTM {
	var s rand.Source
	if len(opts) > 0 && opts[0].Source != nil {
		s = opts[0].Source
	}

	return &LSTM{
		Model: Model{
			Layers: []L.Layer{
				L.LSTM(hiddenSize, L.LSTMOpts{Source: s}),
				L.Linear(outSize, L.LinearOpts{Source: s}),
			},
		},
	}
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
