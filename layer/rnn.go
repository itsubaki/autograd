package layer

import (
	"math"
	"math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

type RNNOpts struct {
	Source rand.Source
}

func RNN(hiddenSize int, opts ...RNNOpts) *RNNT {
	var s rand.Source
	s1 := rand.Uint64N(math.MaxUint64)
	s2 := rand.Uint64N(math.MaxUint64)
	s = rand.NewPCG(s1, s2)

	if len(opts) != 0 && opts[0].Source != nil {
		s = opts[0].Source
	}

	l := make(Layers)
	l.Add("x2h", Linear(hiddenSize, LinearOpts{Source: s}))
	l.Add("h2h", Linear(hiddenSize, LinearOpts{Source: s, InSize: hiddenSize, NoBias: true}))

	return &RNNT{
		Layers: l,
	}
}

type RNNT struct {
	h *variable.Variable
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
