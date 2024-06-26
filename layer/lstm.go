package layer

import (
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

type LSTMOpts struct {
	Source randv2.Source
}

func LSTM(hiddenSize int, opts ...LSTMOpts) *LSTMT {
	var s randv2.Source
	if len(opts) != 0 && opts[0].Source != nil {
		s = opts[0].Source
	}

	l := make(Layers)
	l.Add("x2f", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("x2i", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("x2o", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("x2u", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("h2f", Linear(hiddenSize, LinearOpts{Source: s, InSize: hiddenSize, NoBias: true}))
	l.Add("h2i", Linear(hiddenSize, LinearOpts{Source: s, InSize: hiddenSize, NoBias: true}))
	l.Add("h2o", Linear(hiddenSize, LinearOpts{Source: s, InSize: hiddenSize, NoBias: true}))
	l.Add("h2u", Linear(hiddenSize, LinearOpts{Source: s, InSize: hiddenSize, NoBias: true}))

	return &LSTMT{
		Layers: l,
	}
}

type LSTMT struct {
	h, c *variable.Variable
	Layers
}

func (l *LSTMT) ResetState() {
	l.h = nil
	l.c = nil
}

func (l *LSTMT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LSTMT) Forward(x ...*variable.Variable) []*variable.Variable {
	var f, i, o, u *variable.Variable
	if l.h == nil {
		f = F.Sigmoid(l.Layers["x2f"].First(x...))
		i = F.Sigmoid(l.Layers["x2i"].First(x...))
		o = F.Sigmoid(l.Layers["x2o"].First(x...))
		u = F.Tanh(l.Layers["x2u"].First(x...))
	} else {
		f = F.Sigmoid(F.Add(l.Layers["x2f"].First(x...), l.Layers["h2f"].First(l.h)))
		i = F.Sigmoid(F.Add(l.Layers["x2i"].First(x...), l.Layers["h2i"].First(l.h)))
		o = F.Sigmoid(F.Add(l.Layers["x2o"].First(x...), l.Layers["h2o"].First(l.h)))
		u = F.Tanh(F.Add(l.Layers["x2u"].First(x...), l.Layers["h2u"].First(l.h)))
	}

	var cnew *variable.Variable
	if l.c == nil {
		cnew = F.Mul(i, u) // i * u
	} else {
		cnew = F.Add(F.Mul(f, l.c), F.Mul(i, u)) // f * c + i * u
	}

	hnew := F.Mul(o, F.Tanh(cnew)) // o * tanh(cnew)

	l.h, l.c = hnew, cnew
	return []*variable.Variable{
		hnew,
	}
}
