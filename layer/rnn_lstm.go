package layer

import (
	"math/rand"
	"time"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

type LSTMOpts struct {
	Source rand.Source
}

func LSTM(hiddenSize int, opts ...LSTMOpts) *LSTMT {
	s := rand.NewSource(time.Now().UnixNano())
	if len(opts) != 0 && opts[0].Source != nil {
		s = opts[0].Source
	}

	l := make(Layers)
	l.Add("x2f", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("x2i", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("x2o", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("x2u", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("h2f", Linear(hiddenSize, LinearOpts{Source: s, NoBias: true}))
	l.Add("h2i", Linear(hiddenSize, LinearOpts{Source: s, NoBias: true}))
	l.Add("h2o", Linear(hiddenSize, LinearOpts{Source: s, NoBias: true}))
	l.Add("h2u", Linear(hiddenSize, LinearOpts{Source: s, NoBias: true}))

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
		cnew = F.Mul(i, u)
	} else {
		cnew = F.Add(F.Mul(f, l.c), F.Mul(i, u))
	}

	hnew := F.Mul(o, F.Tanh(cnew))

	l.h, l.c = hnew, cnew
	return []*variable.Variable{
		hnew,
	}
}
