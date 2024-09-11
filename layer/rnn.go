package layer

import (
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

type RNNOptionFunc func(*RNNT)

func WithRNNSource(s randv2.Source) RNNOptionFunc {
	return func(l *RNNT) {
		l.s = s
	}
}
func RNN(hiddenSize int, opts ...RNNOptionFunc) *RNNT {
	rnn := &RNNT{
		Layers: make(Layers),
	}

	for _, opt := range opts {
		opt(rnn)
	}

	rnn.Layers.Add("x2h", Linear(hiddenSize, WithSource(rnn.s)))
	rnn.Layers.Add("h2h", Linear(hiddenSize, WithSource(rnn.s), WithInSize(hiddenSize), WithNoBias()))

	return rnn
}

type RNNT struct {
	h *variable.Variable
	s randv2.Source
	Layers
}

func (l *RNNT) ResetState() {
	l.h = nil
}

func (l *RNNT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *RNNT) Forward(x ...*variable.Variable) []*variable.Variable {
	if l.h == nil {
		l.h = F.Tanh(l.Layers["x2h"].First(x...))
		return []*variable.Variable{
			l.h,
		}
	}

	l.h = F.Tanh(F.Add(l.Layers["x2h"].First(x...), (l.Layers["h2h"].First(l.h))))
	return []*variable.Variable{
		l.h,
	}
}
