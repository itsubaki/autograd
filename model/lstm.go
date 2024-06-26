package model

import (
	randv2 "math/rand/v2"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type LSTMOpts struct {
	Source randv2.Source
}

type LSTM struct {
	Model
}

func NewLSTM(hiddenSize, outSize int, opts ...LSTMOpts) *LSTM {
	var s randv2.Source
	if len(opts) > 0 && opts[0].Source != nil {
		s = opts[0].Source
	}

	layers := []L.Layer{
		L.LSTM(hiddenSize, L.LSTMOpts{Source: s}),
		L.Linear(outSize, L.LinearOpts{Source: s}),
	}

	return &LSTM{
		Model: Model{
			Layers: layers,
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
