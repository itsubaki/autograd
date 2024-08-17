package layer

import (
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

type LSTMOptionFunc func(*LSTMT)

func LSTMWithSource(s randv2.Source) LSTMOptionFunc {
	return func(l *LSTMT) {
		l.s = s
	}
}

func LSTM(hiddenSize int, opts ...LSTMOptionFunc) *LSTMT {
	lstm := &LSTMT{
		Layers: make(Layers),
	}

	for _, opt := range opts {
		opt(lstm)
	}

	lstm.Layers.Add("x2f", Linear(hiddenSize, WithSource(lstm.s)))
	lstm.Layers.Add("x2i", Linear(hiddenSize, WithSource(lstm.s)))
	lstm.Layers.Add("x2o", Linear(hiddenSize, WithSource(lstm.s)))
	lstm.Layers.Add("x2u", Linear(hiddenSize, WithSource(lstm.s)))
	lstm.Layers.Add("h2f", Linear(hiddenSize, WithSource(lstm.s), WithInSize(hiddenSize), WithNoBias()))
	lstm.Layers.Add("h2i", Linear(hiddenSize, WithSource(lstm.s), WithInSize(hiddenSize), WithNoBias()))
	lstm.Layers.Add("h2o", Linear(hiddenSize, WithSource(lstm.s), WithInSize(hiddenSize), WithNoBias()))
	lstm.Layers.Add("h2u", Linear(hiddenSize, WithSource(lstm.s), WithInSize(hiddenSize), WithNoBias()))

	return lstm
}

type LSTMT struct {
	h, c *variable.Variable
	s    randv2.Source
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
