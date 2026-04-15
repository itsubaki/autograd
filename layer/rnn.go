package layer

import (
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

// RNNOptionFunc configures an RNNT layer.
type RNNOptionFunc func(*RNNT)

// WithRNNSource sets the random source used to initialize the layer.
func WithRNNSource(s randv2.Source) RNNOptionFunc {
	return func(l *RNNT) {
		l.s = s
	}
}

// RNN returns a new recurrent neural network layer.
func RNN(hiddenSize int, opts ...RNNOptionFunc) *RNNT {
	rnn := &RNNT{
		Layers: make(Layers),
	}

	for _, opt := range opts {
		opt(rnn)
	}

	rnn.Add("x2h", Linear(hiddenSize, WithSource(rnn.s)))
	rnn.Add("h2h", Linear(hiddenSize, WithSource(rnn.s), WithInSize(hiddenSize), WithNoBias()))

	return rnn
}

// RNNT is a simple recurrent neural network layer.
type RNNT struct {
	h *variable.Variable
	s randv2.Source
	Layers
}

// ResetState clears the hidden state.
func (l *RNNT) ResetState() {
	l.h = nil
}

// First applies the layer and returns the first output.
func (l *RNNT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

// Forward applies the layer to x.
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
